# MusicGen Composer – v3.1 (bug‑fix)
# * FIX: finished signal now always receives mp3 path (str)
# * Single‑30s path now saves audio before emitting
# * Other logic unchanged (30 s or stitched 60 s)

import sys, os, math, torch, numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit,
    QPushButton, QSlider, QFileDialog, QProgressBar, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from scipy.io.wavfile import write
from pydub import AudioSegment
from transformers import AutoProcessor, MusicgenForConditionalGeneration

MODEL_PATH = "models/musicgen-medium"      # local model
TOK_PER_SEC = 74  # empirical: ~74 tokens ≈ 1 second                          # rough ratio
MAX_TOKENS = 1500                           # ≤ 30 s
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def chunk_tokens(sec:int):
    return min(int(sec * TOK_PER_SEC), MAX_TOKENS)

# ──────────────────── Worker ────────────────────
class GenWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)
    error    = pyqtSignal(str)

    def __init__(self, model, proc, prompt, gain, save_path, mode="30s"):
        super().__init__()
        self.model, self.proc = model, proc
        self.prompt, self.gain = prompt, gain
        self.save_path, self.mode = save_path, mode

    # ‑‑ main thread entry
    def run(self):
        try:
            if self.mode == "30s":
                audio_np = self._generate_segment(self.prompt, 30, update=True)
                mp3 = self._save_audio(audio_np, "music_30s")
            else:  # 60 s stitched
                first  = self._generate_segment(self.prompt, 30, update=False)
                cont_p = self.prompt + " – continuation with subtle variation and same tempo"
                second = self._generate_segment(cont_p, 30, update=False)
                self.progress.emit(90)  # generation done
                mp3 = self._save_audio(np.concatenate([first, second]), "music_60s")
            self.progress.emit(100)
            self.finished.emit(mp3)          # ALWAYS str path
        except Exception as e:
            self.error.emit(str(e))

    # generate ≤30 s chunk and return numpy audio
    def _generate_segment(self, prompt, sec, update=True):
        tokens = chunk_tokens(sec)
        inputs = self.proc(text=[prompt], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            audio = self.model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=tokens)
        if update:
            self.progress.emit(50)
        return audio[0].cpu().numpy().squeeze()

    # save wav+mp3, return mp3 path
    def _save_audio(self, audio_np, base):
        audio16 = (np.clip(audio_np * self.gain, -1, 1) * 32767).astype(np.int16)
        sr = self.model.config.audio_encoder.sampling_rate
        wav = os.path.join(self.save_path, f"{base}.wav")
        mp3 = os.path.join(self.save_path, f"{base}.mp3")
        write(wav, sr, audio16)
        AudioSegment.from_wav(wav).export(mp3, format="mp3", bitrate="192k")
        return mp3

# ──────────────────── GUI ────────────────────
class MusicGenGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MusicGen Composer v3.1")
        self.resize(620, 480)
        self.save_path = os.getcwd()
        self._build_ui(); self._load_model(); self._dark()

    def _build_ui(self):
        L = QVBoxLayout(self)
        # prompt
        L.addWidget(QLabel("Prompt:"))
        self.prompt = QTextEdit("A minimalist solo piano instrumental with a soft, repetitive melody and calm atmosphere, perfect for deep focus and Pomodoro sessions.")
        L.addWidget(self.prompt)
        # gain
        gbox = QHBoxLayout(); self.glbl = QLabel("Gain: 1.0×"); self.gain = QSlider(Qt.Horizontal)
        self.gain.setRange(10, 30); self.gain.setValue(18); self.gain.valueChanged.connect(lambda v: self.glbl.setText(f"Gain: {v/10:.1f}×"))
        gbox.addWidget(self.glbl); gbox.addWidget(self.gain); L.addLayout(gbox)
        # buttons
        b = QHBoxLayout()
        self.folder = QPushButton("Save Folder…"); self.folder.clicked.connect(self._choose)
        self.btn30 = QPushButton("Generate 30s"); self.btn60 = QPushButton("Generate 60s (2×30)")
        self.btn30.clicked.connect(lambda: self._start("30s")); self.btn60.clicked.connect(lambda: self._start("60s"))
        b.addWidget(self.folder); b.addWidget(self.btn30); b.addWidget(self.btn60); L.addLayout(b)
        # progress / status
        self.prog = QProgressBar(); L.addWidget(self.prog); self.status = QLabel("Ready ✅"); L.addWidget(self.status)

    def _dark(self):
        self.setStyleSheet("""QWidget{background:#222;color:#eee;font-family:Arial}
        QTextEdit{background:#333;color:#eee;border:1px solid #555}
        QPushButton{background:#2d8cf0;color:white;border:none;padding:8px;border-radius:4px}
        QPushButton:hover{background:#1e6fb3}
        QSlider::groove:horizontal{background:#444;height:4px}
        QSlider::handle:horizontal{background:#2ecc71;width:14px;margin:-6px 0;border-radius:7px}
        QProgressBar{background:#555;text-align:center;border-radius:3px}
        QProgressBar::chunk{background:#2ecc71}""")

    def _load_model(self):
        self.status.setText("Loading model…"); QApplication.processEvents()
        self.model = MusicgenForConditionalGeneration.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).to("cuda" if torch.cuda.is_available() else "cpu")
        self.proc  = AutoProcessor.from_pretrained(MODEL_PATH)
        self.status.setText("Model ready ✅")

    def _choose(self):
        d = QFileDialog.getExistingDirectory(self, "Select Folder", self.save_path)
        if d: self.save_path = d; self.status.setText(f"Save → {d}")

    def _start(self, m):
        txt = self.prompt.toPlainText().strip()
        if not txt: QMessageBox.warning(self, "Prompt", "Prompt cannot be empty"); return
        g = self.gain.value()/10; self.prog.setValue(0); self.status.setText("Generating…")
        self.worker = GenWorker(self.model, self.proc, txt, g, self.save_path, m)
        self.worker.progress.connect(self.prog.setValue); self.worker.finished.connect(self._done); self.worker.error.connect(self._err); self.worker.start()

    def _done(self, p):
        self.status.setText(f"Saved → {p}"); QMessageBox.information(self, "Done", f"Music saved:\n{p}")

    def _err(self, msg):
        self.status.setText("Error ❌"); QMessageBox.critical(self, "Error", msg)

if __name__ == "__main__":
    app = QApplication(sys.argv); gui = MusicGenGUI(); gui.show(); sys.exit(app.exec_())
