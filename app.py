"""
Voice Flow — push-to-talk dictation overlay in the spirit of Wispr Flow.

Hold Right Ctrl and speak. Release to transcribe; the text is pasted straight
into whatever field had focus. Tap Right Ctrl quickly instead of holding to
latch hands-free mode, then tap again (or stay silent) to finish.

    pip install PySide6 faster-whisper sounddevice numpy pyperclip pyautogui pynput

macOS: grant Accessibility + Microphone permission to your terminal/Python.
Linux (X11): needs xclip or xsel for the clipboard.
"""

from __future__ import annotations

import math
import os
import queue
import re
import signal
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pyperclip
import sounddevice as sd
from pynput import keyboard

# pyautogui claims Windows DPI awareness the moment it is imported, which locks
# Qt out of setting its own. It is imported lazily, after QApplication exists.
_AUTOGUI = None


def autogui():
    global _AUTOGUI
    if _AUTOGUI is None:
        import pyautogui as _pg
        _pg.FAILSAFE = False
        _pg.PAUSE = 0
        _AUTOGUI = _pg
    return _AUTOGUI


from PySide6.QtCore import (
    QEasingCurve, QPoint, QRect, QPropertyAnimation, QRectF, QSettings, QTimer,
    Qt, QObject, Signal,
)
from PySide6.QtGui import (
    QAction, QColor, QFont, QFontMetrics, QIcon, QPainter, QPainterPath, QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication, QFrame, QHBoxLayout, QLabel, QMenu, QStackedLayout,
    QSystemTrayIcon, QVBoxLayout, QWidget,
)


# In a windowed (no-console) build — e.g. the packaged .exe — sys.stdout and
# sys.stderr are None, so any print() or library progress bar (huggingface's
# tqdm) would crash with AttributeError. Route them to a throwaway sink.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")


# ──────────────────────────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # audio
    sample_rate: int = 16000
    channels: int = 1

    # model
    model_size: str = "small.en"      # tiny.en / base.en / small.en / medium / large-v3
    language: str | None = "en"       # None = auto-detect (also drops the .en model)
    beam_size: int = 1                # 1 = fastest, 5 = most accurate

    # hotkeys  — never bind a letter key; it types into your documents.
    ptt_key: object = keyboard.Key.ctrl_r
    tap_to_latch_seconds: float = 0.35   # press shorter than this = hands-free
    cancel_key: object = keyboard.Key.esc
    quit_double_tap_seconds: float = 0.6   # tap Esc twice this fast to quit

    # behaviour
    auto_send: bool = False              # press Enter after pasting
    restore_clipboard: bool = True       # put your old clipboard back
    min_record_seconds: float = 0.35     # ignore accidental taps
    max_record_seconds: float = 180.0
    silence_stop_seconds: float = 2.5    # auto-finish latched mode after silence

    # text polish
    remove_fillers: bool = True
    trailing_space: bool = True
    vocabulary: tuple = ("Claude", "Anthropic", "PySide6", "Whisper", "GitHub", "API")

    # window
    always_expanded: bool = False        # keep the pill in its expanded form at rest
    translucent: bool = True             # set False if the overlay flickers on Windows


CFG = Config()

# One accent, one surface, one set of state hues. Nothing else gets a colour.
ACCENT = "#818cf8"
SURFACE = QColor(16, 16, 20, 242)
BORDER = QColor(255, 255, 255, 28)

SHADOW_PAD = 18      # window padding reserved for the hand-painted shadow
SHADOW_STEPS = 7     # concentric passes; more = softer, slightly slower

STATE_COLOR = {
    "loading":     "#52525b",
    "downloading": "#38bdf8",
    "idle":        "#52525b",
    "listening":   "#34d399",
    "latched":     "#38bdf8",
    "working":     "#fbbf24",
    "done":        ACCENT,
    "error":       "#fb7185",
}

FILLERS = {"um", "uh", "erm", "hmm", "mm", "uhh", "umm", "ah", "eh"}


def ui_font(size: int, bold: bool = False) -> QFont:
    families = ["Inter", "SF Pro Text", "Segoe UI Variable", "Segoe UI", "Ubuntu", "Sans Serif"]
    f = QFont()
    f.setFamilies(families)
    f.setPointSize(size)
    f.setWeight(QFont.Weight.DemiBold if bold else QFont.Weight.Normal)
    return f


# ──────────────────────────────────────────────────────────────────────────────
#  Text polish
# ──────────────────────────────────────────────────────────────────────────────

def polish(text: str, cfg: Config) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    if cfg.remove_fillers:
        words = [w for w in text.split() if w.strip(".,!?;:").lower() not in FILLERS]
        text = " ".join(words)
    for term in cfg.vocabulary:
        text = re.sub(rf"\b{re.escape(term)}\b", term, text, flags=re.IGNORECASE)
    if text:
        text = text[0].upper() + text[1:]
    if cfg.trailing_space:
        text += " "
    return text


# ──────────────────────────────────────────────────────────────────────────────
#  Model storage + first-run download progress
# ──────────────────────────────────────────────────────────────────────────────

def model_dir() -> str:
    """Where downloaded speech models live — under %LOCALAPPDATA% on Windows."""
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    path = os.path.join(base, "VoiceFlow", "models")
    os.makedirs(path, exist_ok=True)
    return path


class _DownloadProgress:
    """Aggregates byte counts across huggingface_hub's per-file progress bars
    into a single 0..1 fraction, reported through a callback."""

    def __init__(self, callback):
        self.callback = callback
        self.total = 0
        self.done = 0
        self.lock = threading.Lock()

    def tqdm_class(self):
        prog = self
        from huggingface_hub.utils import tqdm as hf_tqdm

        class _Reporting(hf_tqdm):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self._prev = 0
                with prog.lock:
                    if self.total:
                        prog.total += self.total

            def update(self, n=1):
                result = super().update(n)
                with prog.lock:
                    prog.done += (self.n - self._prev)
                    self._prev = self.n
                    if prog.total:
                        prog.callback(min(1.0, prog.done / prog.total))
                return result

        return _Reporting


# ──────────────────────────────────────────────────────────────────────────────
#  Audio + transcription engine
# ──────────────────────────────────────────────────────────────────────────────

class Signals(QObject):
    model_ready = Signal()
    summon = Signal()
    quit_requested = Signal()
    listening = Signal(bool)      # True = latched / hands-free
    working = Signal()
    done = Signal(str, float)     # text, seconds of audio
    cancelled = Signal()
    error = Signal(str)
    hotkey_captured = Signal(object)   # a rebind finished (key, or None if cancelled)
    downloading = Signal(float)        # first-run model fetch: 0..1, or <0 = indeterminate
    download_failed = Signal(str)      # fetch failed; the app offers a retry


class VoiceEngine:
    """Records on demand, transcribes on a background worker, pastes the result."""

    def __init__(self, cfg: Config, signals: Signals):
        self.cfg = cfg
        self.signals = signals
        self.recording = False
        self.latched = False
        self.model = None

        self._lock = threading.Lock()
        self._chunks: list[np.ndarray] = []
        self._stream = None
        self._started_at = 0.0
        self._last_voice_at = 0.0

        self.awaiting_retry = False        # True while a failed download waits for the user
        self._retry = threading.Event()
        self._dl_err = ""

        self.level = 0.0          # 0..1, polled by the UI at frame rate
        self._jobs: "queue.Queue[np.ndarray]" = queue.Queue()

        threading.Thread(target=self._load_model, daemon=True).start()
        threading.Thread(target=self._worker, daemon=True).start()

    # ---- model -------------------------------------------------------------
    def _load_model(self):
        size = self.cfg.model_size
        if size.endswith(".en") and self.cfg.language not in (None, "en"):
            size = size[:-3]

        root = model_dir()

        # 1. already downloaded into our app cache?
        if self._attempt_load(size, root):
            self.signals.model_ready.emit()
            return
        # 2. already sitting in the user's default HuggingFace cache (dev machines)?
        if self._attempt_load(size, None):
            self.signals.model_ready.emit()
            return
        # 3. fetch it, retrying as many times as the user asks
        while not self._download(size, root):
            self.awaiting_retry = True
            self.signals.download_failed.emit(self._dl_err)
            self._retry.wait()
            self._retry.clear()
            self.awaiting_retry = False

        if self._attempt_load(size, root):
            self.signals.model_ready.emit()
        else:
            self.signals.error.emit("Could not load the speech model")

    def _attempt_load(self, size: str, root: str | None) -> bool:
        """Load from local files only (no network). Returns True on success."""
        from faster_whisper import WhisperModel
        for device, compute in (("cuda", "float16"), ("cpu", "int8")):
            try:
                self.model = WhisperModel(
                    size, device=device, compute_type=compute,
                    download_root=root, local_files_only=True,
                )
                print(f"[voice-flow] model '{size}' loaded on {device}"
                      + (f" from {root}" if root else " from default cache"))
                return True
            except Exception:
                continue
        return False

    def _repo_id(self, size: str) -> str:
        try:
            from faster_whisper.utils import _MODELS
            repo = _MODELS.get(size)
            if repo:
                return repo
        except Exception:
            pass
        return "Systran/faster-whisper-" + size

    def _download(self, size: str, root: str) -> bool:
        """Fetch the model into `root` with progress. Returns False on failure."""
        try:
            from huggingface_hub import snapshot_download
            repo = self._repo_id(size)
            print(f"[voice-flow] downloading model '{size}' from {repo}")
            self.signals.downloading.emit(0.0)
            prog = _DownloadProgress(lambda f: self.signals.downloading.emit(f))
            try:
                snapshot_download(repo, cache_dir=root, tqdm_class=prog.tqdm_class())
            except TypeError:
                # older huggingface_hub without tqdm_class → show an indeterminate bar
                self.signals.downloading.emit(-1.0)
                snapshot_download(repo, cache_dir=root)
            return True
        except Exception as exc:
            self._dl_err = str(exc)[:70]
            print(f"[voice-flow] download failed: {exc}")
            return False

    def retry(self):
        """Called from the UI to re-attempt a failed download."""
        self._retry.set()

    @property
    def ready(self) -> bool:
        return self.model is not None

    # ---- capture -----------------------------------------------------------
    def _callback(self, indata, frames, time_info, status):
        block = indata.copy()
        rms = float(np.sqrt(np.mean(np.square(block))) + 1e-9)
        db = 20.0 * math.log10(rms)
        self.level = max(0.0, min(1.0, (db + 58.0) / 42.0))   # -58dB..-16dB → 0..1
        with self._lock:
            if not self.recording:
                return
            self._chunks.append(block)
            if self.level > 0.16:
                self._last_voice_at = time.time()

    def start(self, latched: bool = False):
        if not self.ready:
            return
        with self._lock:
            if self.recording:
                return
            self._chunks = []
            self.recording = True
        self.latched = latched
        self._started_at = self._last_voice_at = time.time()
        try:
            self._stream = sd.InputStream(
                samplerate=self.cfg.sample_rate,
                channels=self.cfg.channels,
                dtype="float32",
                blocksize=512,
                callback=self._callback,
            )
            self._stream.start()
            self.signals.listening.emit(latched)
        except Exception as exc:
            with self._lock:
                self.recording = False
            self.signals.error.emit(f"No microphone: {exc}")

    def _close_stream(self):
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        self.level = 0.0

    def stop(self, cancel: bool = False):
        with self._lock:
            if not self.recording:
                return
            self.recording = False
            chunks = self._chunks
            self._chunks = []
        self.latched = False
        self._close_stream()

        duration = len(chunks) * 512 / self.cfg.sample_rate if chunks else 0.0
        if cancel or duration < self.cfg.min_record_seconds:
            self.signals.cancelled.emit()
            return
        self.signals.working.emit()
        self._jobs.put(np.concatenate(chunks, axis=0).flatten())

    # ---- timing helpers polled by the UI ----------------------------------
    def elapsed(self) -> float:
        return time.time() - self._started_at if self.recording else 0.0

    def silent_for(self) -> float:
        return time.time() - self._last_voice_at if self.recording else 0.0

    # ---- transcription -----------------------------------------------------
    def _worker(self):
        while True:
            audio = self._jobs.get()
            seconds = len(audio) / self.cfg.sample_rate
            try:
                segments, _ = self.model.transcribe(
                    audio,
                    language=self.cfg.language,
                    beam_size=self.cfg.beam_size,
                    vad_filter=True,
                    vad_parameters={"min_silence_duration_ms": 300},
                    condition_on_previous_text=False,
                    initial_prompt=", ".join(self.cfg.vocabulary),
                )
                text = polish(" ".join(s.text.strip() for s in segments), self.cfg)
                if not text.strip():
                    self.signals.error.emit("Nothing heard — try speaking closer")
                    continue
                self._paste(text)
                self.signals.done.emit(text.strip(), seconds)
            except Exception as exc:
                self.signals.error.emit(str(exc)[:70])

    def _paste(self, text: str):
        previous = None
        if self.cfg.restore_clipboard:
            try:
                previous = pyperclip.paste()
            except Exception:
                previous = None
        try:
            pyperclip.copy(text)
        except Exception as exc:
            raise RuntimeError(f"Clipboard unavailable: {exc}")

        time.sleep(0.06)
        pg = autogui()
        modifier = "command" if sys.platform == "darwin" else "ctrl"
        pg.hotkey(modifier, "v")
        if self.cfg.auto_send:
            time.sleep(0.08)
            pg.press("enter")

        if previous is not None:
            def restore():
                time.sleep(0.6)
                try:
                    pyperclip.copy(previous)
                except Exception:
                    pass
            threading.Thread(target=restore, daemon=True).start()


# ──────────────────────────────────────────────────────────────────────────────
#  Waveform — the one place this design spends its boldness
# ──────────────────────────────────────────────────────────────────────────────

class Waveform(QWidget):
    BARS = 24

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(22)
        self.setMinimumWidth(90)
        self.history = deque([0.0] * self.BARS, maxlen=self.BARS)
        self.shown = [0.0] * self.BARS
        self.mode = "idle"
        self.color = QColor(STATE_COLOR["idle"])
        self.phase = 0.0

    def set_mode(self, mode: str):
        self.mode = mode
        self.color = QColor(STATE_COLOR.get(mode, ACCENT))
        if mode not in ("listening", "latched"):
            self.history = deque([0.0] * self.BARS, maxlen=self.BARS)
        self.update()

    def push(self, level: float):
        self.history.append(level)

    def tick(self):
        self.phase += 0.22
        if self.mode in ("listening", "latched"):
            targets = list(self.history)
        elif self.mode == "working":
            targets = [
                0.18 + 0.55 * max(0.0, math.sin(self.phase - i * 0.32)) ** 2
                for i in range(self.BARS)
            ]
        elif self.mode == "done":
            targets = [0.5] * self.BARS
        else:
            base = 0.03 + 0.02 * math.sin(self.phase * 0.25)
            targets = [base] * self.BARS

        for i, t in enumerate(targets):
            ease = 0.6 if t > self.shown[i] else 0.18    # fast attack, slow release
            self.shown[i] += (t - self.shown[i]) * ease
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)

        n = self.BARS
        gap = 3.0
        bw = max(2.0, (self.width() - gap * (n - 1)) / n)
        mid = self.height() / 2
        peak = self.height() - 4

        for i, v in enumerate(self.shown):
            h = max(3.0, min(peak, v * peak))
            x = i * (bw + gap)
            # newest sample on the right stays brightest
            alpha = 90 + int(150 * (i + 1) / n) if self.mode in ("listening", "latched") else 210
            c = QColor(self.color)
            c.setAlpha(min(255, alpha))
            p.setBrush(c)
            p.drawRoundedRect(QRectF(x, mid - h / 2, bw, h), bw / 2, bw / 2)
        p.end()


class StatusDot(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(12, 12)
        self.color = QColor(STATE_COLOR["idle"])
        self.pulsing = False
        self.phase = 0.0

    def set_state(self, state: str):
        self.color = QColor(STATE_COLOR.get(state, ACCENT))
        self.pulsing = state in ("listening", "latched", "working")
        self.update()

    def tick(self):
        if self.pulsing:
            self.phase += 0.16
            self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        if self.pulsing:
            glow = QColor(self.color)
            glow.setAlpha(int(70 + 60 * (0.5 + 0.5 * math.sin(self.phase))))
            p.setBrush(glow)
            p.drawEllipse(0, 0, 12, 12)
        p.setBrush(self.color)
        p.drawEllipse(3, 3, 6, 6)
        p.end()


# ──────────────────────────────────────────────────────────────────────────────
#  Overlay
# ──────────────────────────────────────────────────────────────────────────────

class Overlay(QWidget):
    REST_W = 76          # at rest the pill is just the status dot — a tiny capsule
    ACTIVE_W = 320       # widens to reveal the waveform + timer while active
    CARD_H = 34          # pill height; the window adds SHADOW_PAD around it

    def __init__(self, cfg: Config, engine: VoiceEngine, on_toggle=None, on_menu=None):
        super().__init__()
        self.cfg = cfg
        self.engine = engine
        self.on_toggle = on_toggle          # called on a genuine click (not a drag)
        self.on_menu = on_menu              # called with a global QPoint on right-click
        self.state = "loading"

        self._drag_from: QPoint | None = None
        self._press_pos: QPoint | None = None
        self._press_time = 0.0
        self._dragging = False

        self.setWindowTitle("Voice Flow")
        self.setWindowFlags(
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        if cfg.translucent:
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        win_h = self.CARD_H + 2 * SHADOW_PAD
        self.setFixedHeight(win_h)          # height is constant; only the width breathes
        self.resize(self.REST_W, win_h)

        self._build()
        self._restore_position()

        self.fade = QPropertyAnimation(self, b"windowOpacity", self)
        self.fade.setDuration(180)
        self.fade.setEasingCurve(QEasingCurve.Type.InOutCubic)

        # The pill grows/shrinks by animating its geometry so it stays centred.
        self.geo = QPropertyAnimation(self, b"geometry", self)
        self.geo.setDuration(200)
        self.geo.setEasingCurve(QEasingCurve.Type.OutQuint)

        self.frame_timer = QTimer(self)
        self.frame_timer.timeout.connect(self._frame)
        self.frame_timer.start(16)

        self.set_state("loading")

    # ---- construction ------------------------------------------------------
    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(SHADOW_PAD, SHADOW_PAD, SHADOW_PAD, SHADOW_PAD)

        # No QGraphicsDropShadowEffect: on Windows its blur extends past the
        # window rect, and layered (translucent) windows reject repaint regions
        # with negative coordinates. The shadow is painted by hand instead.
        self.card = QFrame(self)
        self.card.setObjectName("card")
        self.card.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        outer.addWidget(self.card)

        row = QHBoxLayout(self.card)
        row.setContentsMargins(14, 6, 14, 6)
        row.setSpacing(10)

        self.dot = StatusDot()

        # The centre shows the live waveform or a text message, never both — a
        # QStackedLayout swaps between them so the pill stays a single row.
        self.center = QWidget()
        self.center_stack = QStackedLayout(self.center)
        self.center_stack.setContentsMargins(0, 0, 0, 0)
        self.wave = Waveform()
        self.message = QLabel("")
        self.message.setFont(ui_font(9))
        self.message.setStyleSheet("color:#e4e4e7;")
        self.message.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        self.center_stack.addWidget(self.wave)
        self.center_stack.addWidget(self.message)

        self.timer_label = QLabel("")
        self.timer_label.setFont(ui_font(9))
        self.timer_label.setStyleSheet(f"color:{ACCENT};")
        self.timer_label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)

        row.addWidget(self.dot)
        row.addWidget(self.center, 1)
        row.addWidget(self.timer_label)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.card.geometry())
        radius = r.height() / 2      # fully rounded ends → a pill

        if self.cfg.translucent:
            p.setPen(Qt.PenStyle.NoPen)
            for i in range(SHADOW_STEPS, 0, -1):
                spread = i * (SHADOW_PAD - 4) / SHADOW_STEPS
                p.setBrush(QColor(0, 0, 0, max(4, int(40 / i))))
                p.drawRoundedRect(
                    r.adjusted(-spread, -spread + 3, spread, spread + 3),
                    radius + spread * 0.4, radius + spread * 0.4,
                )

        path = QPainterPath()
        path.addRoundedRect(r, radius, radius)
        p.fillPath(path, SURFACE)
        p.setPen(QPen(BORDER, 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)
        p.end()

    # ---- state -------------------------------------------------------------
    def set_state(self, state: str, message: str | None = None, timer: str = ""):
        self.state = state
        self.dot.set_state(state)
        self.wave.set_mode(state)

        # done / error / downloading speak in words; other states use the waveform.
        if state in ("done", "error", "downloading") and message:
            self.message.setText(self._elide(message))
            self.center_stack.setCurrentWidget(self.message)
        else:
            self.center_stack.setCurrentWidget(self.wave)
        self.setToolTip(message or "")

        self.timer_label.setText(timer)

        expanded = self.cfg.always_expanded or state in (
            "listening", "latched", "working", "done", "error", "downloading",
        )
        # At rest the pill is just the status dot; the waveform + timer appear
        # only once it expands.
        self.center.setVisible(expanded)
        self.timer_label.setVisible(expanded)
        self._resize_to(self.ACTIVE_W if expanded else self.REST_W)
        self._sync_timer()

    def flash_message(self, text: str, msecs: int = 4500):
        """Briefly show a one-off message (e.g. the first-run primer), then rest."""
        self.message.setText(self._elide(text))
        self.center_stack.setCurrentWidget(self.message)
        self._resize_to(self.ACTIVE_W)
        QTimer.singleShot(
            msecs,
            lambda: self.set_state("idle") if self.state == "idle" else None,
        )

    def set_hint(self, text: str):
        self.setToolTip(text)

    def _elide(self, text: str) -> str:
        metrics = QFontMetrics(self.message.font())
        avail = self.ACTIVE_W - 2 * SHADOW_PAD - 28 - 64   # margins, dot, timer
        return metrics.elidedText(text, Qt.TextElideMode.ElideRight, max(80, avail))

    # ---- visibility --------------------------------------------------------
    def _sync_timer(self):
        """Full frame rate only while something is actually moving."""
        busy = self.state in ("listening", "latched", "working", "done")
        self.frame_timer.start(16 if busy else 90)

    def reveal(self):
        """Bring the pill to front (used at startup and from the tray/summon)."""
        if not self.isVisible():
            self.setWindowOpacity(0.0)
            self.show()
            self.fade.stop()
            self.fade.setStartValue(0.0)
            self.fade.setEndValue(1.0)
            self.fade.start()
        else:
            self.setWindowOpacity(1.0)
        self.raise_()
        self._sync_timer()

    # ---- size ---------------------------------------------------------------
    def _target_geometry(self, width: int) -> QRect:
        g = self.geometry()
        cx = g.center().x()                 # grow / shrink about the centre
        return QRect(int(cx - width / 2), g.y(), width, g.height())

    def _resize_to(self, width: int):
        target = self._target_geometry(width)
        if self.geometry() == target:
            return
        self.geo.stop()
        self.geo.setStartValue(self.geometry())
        self.geo.setEndValue(target)
        self.geo.start()

    def reset_position(self):
        """Snap the pill back to a visible default — recovery for an off-screen pill."""
        area = QApplication.primaryScreen().availableGeometry()
        self.move(area.center().x() - self.width() // 2,
                  area.bottom() - self.height() - 40)
        QSettings("VoiceFlow", "overlay").setValue("pos", self.pos())

    # ---- frame loop --------------------------------------------------------
    def _frame(self):
        if self.state in ("listening", "latched"):
            self.wave.push(self.engine.level)
            secs = self.engine.elapsed()
            self.timer_label.setText(f"{int(secs) // 60}:{int(secs) % 60:02d}")
        self.wave.tick()
        self.dot.tick()

    # ---- mouse: click toggles, drag moves ----------------------------------
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_pos = event.globalPosition().toPoint()
            self._press_time = time.time()
            self._drag_from = self._press_pos - self.pos()
            self._dragging = False

    def mouseMoveEvent(self, event):
        if self._drag_from is None:
            return
        gp = event.globalPosition().toPoint()
        if not self._dragging and (gp - self._press_pos).manhattanLength() > 4:
            self._dragging = True          # crossed the slop threshold → it's a drag
        if self._dragging:
            self.geo.stop()                # cancel any in-flight expand while dragging
            self.move(gp - self._drag_from)

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton or self._drag_from is None:
            return
        was_drag = self._dragging
        held = time.time() - self._press_time
        self._drag_from = None
        self._dragging = False
        if was_drag:
            QSettings("VoiceFlow", "overlay").setValue("pos", self.pos())
        elif held < 0.25 and self.on_toggle is not None:
            self.on_toggle()               # a real click: start / stop recording

    def contextMenuEvent(self, event):
        if self.on_menu is not None:
            self.on_menu(event.globalPos())

    def _restore_position(self):
        saved = QSettings("VoiceFlow", "overlay").value("pos")
        if isinstance(saved, QPoint):
            self.move(saved)
            return
        area = QApplication.primaryScreen().availableGeometry()
        self.move(area.center().x() - self.width() // 2, area.bottom() - self.height() - 40)


# ──────────────────────────────────────────────────────────────────────────────
#  Hotkey (de)serialisation + Windows startup registration
# ──────────────────────────────────────────────────────────────────────────────

_KEY_PRETTY = {
    "ctrl_r": "Right Ctrl", "ctrl_l": "Left Ctrl",
    "alt_r": "Right Alt", "alt_l": "Left Alt", "alt_gr": "Right Alt",
    "shift_r": "Right Shift", "shift_l": "Left Shift",
    "cmd": "Cmd", "cmd_r": "Right Cmd",
}


def key_name(key) -> str:
    """A human-friendly label for a pynput key, e.g. 'Right Ctrl' or 'F9'."""
    if isinstance(key, keyboard.Key):
        return _KEY_PRETTY.get(key.name, key.name.replace("_", " ").title())
    if isinstance(key, keyboard.KeyCode):
        if key.char:
            return key.char.upper()
        return f"vk{key.vk}"
    return str(key)


def key_to_str(key) -> str:
    """Serialise a pynput key so it can live in QSettings."""
    if isinstance(key, keyboard.Key):
        return key.name
    if isinstance(key, keyboard.KeyCode):
        if key.char is not None:
            return f"char:{key.char}"
        if key.vk is not None:
            return f"vk:{key.vk}"
    return ""


def str_to_key(s: str):
    """Inverse of key_to_str; returns None if the string is empty/unknown."""
    if not s:
        return None
    if s.startswith("char:"):
        return keyboard.KeyCode.from_char(s[5:])
    if s.startswith("vk:"):
        try:
            return keyboard.KeyCode.from_vk(int(s[3:]))
        except ValueError:
            return None
    return getattr(keyboard.Key, s, None)


APP_RUN_NAME = "VoiceFlow"
_RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"


def _startup_command() -> str:
    """The command Windows should run at logon."""
    if getattr(sys, "frozen", False):          # a PyInstaller-built .exe
        return f'"{sys.executable}"'
    return f'"{sys.executable}" "{os.path.abspath(__file__)}"'


def is_run_on_startup() -> bool:
    if not sys.platform.startswith("win"):
        return False
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _RUN_KEY) as k:
            winreg.QueryValueEx(k, APP_RUN_NAME)
            return True
    except OSError:
        return False


def set_run_on_startup(enable: bool):
    if not sys.platform.startswith("win"):
        return
    import winreg
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _RUN_KEY, 0, winreg.KEY_SET_VALUE) as k:
        if enable:
            winreg.SetValueEx(k, APP_RUN_NAME, 0, winreg.REG_SZ, _startup_command())
        else:
            try:
                winreg.DeleteValue(k, APP_RUN_NAME)
            except FileNotFoundError:
                pass


# ──────────────────────────────────────────────────────────────────────────────
#  Hotkeys
# ──────────────────────────────────────────────────────────────────────────────

class Hotkeys:
    """Hold to talk. A quick tap latches hands-free mode; tap again to finish.

    Esc cancels a recording, or summons the overlay when nothing is running.
    Two quick taps of Esc quits the app from anywhere — the overlay hides
    itself when idle, so there has to be a way out that does not need a click.
    """

    def __init__(self, cfg: Config, engine: VoiceEngine, on_cancel, on_summon, on_quit):
        self.cfg = cfg
        self.engine = engine
        self.on_cancel = on_cancel
        self.on_summon = on_summon
        self.on_quit = on_quit
        self.pressed_at = 0.0
        self.latched = False
        self.esc_down = False        # blocks key auto-repeat from faking a double tap
        self.esc_at = 0.0
        self._capture_cb = None      # set while rebinding the hotkey
        self.listener = keyboard.Listener(on_press=self._press, on_release=self._release)
        self.listener.start()

    def capture_next(self, callback):
        """Grab the next key press as the new hotkey. Esc cancels; printable keys
        are ignored (they'd type into documents). Fires callback(key|None)."""
        self._capture_cb = callback

    def _press(self, key):
        if self._capture_cb is not None:
            if key == self.cfg.cancel_key:
                cb, self._capture_cb = self._capture_cb, None
                cb(None)
                return
            # ignore letters/digits — binding one would type into your documents
            if isinstance(key, keyboard.KeyCode) and key.char and key.char.isalnum():
                return
            cb, self._capture_cb = self._capture_cb, None
            self.cfg.ptt_key = key       # apply at once so the listener uses it
            cb(key)
            return

        if key == self.cfg.cancel_key:
            if self.esc_down:
                return
            self.esc_down = True
            now = time.time()
            if now - self.esc_at < self.cfg.quit_double_tap_seconds:
                self.esc_at = 0.0
                self.on_quit()
                return
            self.esc_at = now
            if self.engine.recording:
                self.latched = False
                self.on_cancel()
            else:
                self.on_summon()
            return

        if key != self.cfg.ptt_key or self.latched:
            return
        if not self.engine.recording:
            self.pressed_at = time.time()
            self.engine.start()

    def _release(self, key):
        if key == self.cfg.cancel_key:
            self.esc_down = False
            return
        if key != self.cfg.ptt_key:
            return
        if self.latched:
            self.latched = False
            self.engine.stop()
            return
        if time.time() - self.pressed_at < self.cfg.tap_to_latch_seconds:
            self.latched = True
            self.engine.latched = True
        else:
            self.engine.stop()

    def clear_latch(self):
        self.latched = False

    def stop(self):
        try:
            self.listener.stop()
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
#  App
# ──────────────────────────────────────────────────────────────────────────────

class App:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._load_settings()
        self.qt = QApplication(sys.argv)
        self.qt.setQuitOnLastWindowClosed(False)
        autogui()   # safe now: Qt already owns the DPI awareness context

        for sig in (signal.SIGINT, getattr(signal, "SIGTERM", None)):
            if sig is not None:
                try:
                    signal.signal(sig, lambda *_: self.signals.quit_requested.emit())
                except (ValueError, OSError):
                    pass

        self.signals = Signals()
        self.engine = VoiceEngine(cfg, self.signals)
        self.window = Overlay(
            cfg, self.engine,
            on_toggle=self._toggle_record,
            on_menu=self._show_context_menu,
        )
        self.window.reveal()
        self.hotkeys = Hotkeys(
            cfg, self.engine,
            on_cancel=self._cancel,
            on_summon=self.signals.summon.emit,
            on_quit=self.signals.quit_requested.emit,
        )
        self.last_text = ""

        self.signals.model_ready.connect(self._on_ready)
        self.signals.summon.connect(self.window.reveal)
        self.signals.quit_requested.connect(self.shutdown)
        self.signals.listening.connect(self._on_listening)
        self.signals.working.connect(self._on_working)
        self.signals.done.connect(self._on_done)
        self.signals.cancelled.connect(self._on_cancelled)
        self.signals.error.connect(self._on_error)
        self.signals.hotkey_captured.connect(self._on_hotkey_captured)
        self.signals.downloading.connect(self._on_downloading)
        self.signals.download_failed.connect(self._on_download_failed)

        self.guard = QTimer()
        self.guard.timeout.connect(self._guard)
        self.guard.start(200)

        self._build_tray()
        print("[voice-flow] hold Right Ctrl to dictate • tap it for hands-free • Esc cancels")

    # ---- settings ----------------------------------------------------------
    def _load_settings(self):
        """Restore choices persisted from a previous run (hotkey, auto-send)."""
        s = QSettings("VoiceFlow", "settings")
        auto = s.value("auto_send")
        if auto is not None:
            self.cfg.auto_send = auto in (True, "true", "True", 1, "1")
        key = str_to_key(s.value("ptt_key") or "")
        if key is not None:
            self.cfg.ptt_key = key

    # ---- tray --------------------------------------------------------------
    def _build_tray(self):
        pix = QPixmap(64, 64)
        pix.fill(Qt.GlobalColor.transparent)
        p = QPainter(pix)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(ACCENT))
        for i, h in enumerate((18, 34, 48, 34, 18)):
            p.drawRoundedRect(QRectF(8 + i * 11, 32 - h / 2, 7, h), 3.5, 3.5)
        p.end()

        self.tray = QSystemTrayIcon(QIcon(pix))
        self.tray_menu = QMenu()
        self._populate_menu(self.tray_menu)
        self.tray.setContextMenu(self.tray_menu)
        self.tray.setToolTip("Voice Flow — hold Right Ctrl to dictate")
        self.tray.activated.connect(lambda _: self.window.reveal())
        self.tray.show()

    # ---- shared menu (tray + pill right-click) ------------------------------
    def _populate_menu(self, menu: QMenu):
        """Fill a menu with the app's actions. Used by both the tray and the pill."""
        show = QAction("Show overlay", menu)
        show.triggered.connect(self.window.reveal)

        copy_last = QAction("Copy last transcript", menu)
        copy_last.triggered.connect(lambda: pyperclip.copy(self.last_text))
        copy_last.setEnabled(bool(self.last_text))

        send = QAction("Press Enter after pasting", menu, checkable=True)
        send.setChecked(self.cfg.auto_send)
        send.toggled.connect(self._set_auto_send)

        hotkey = QAction(f"Change hotkey…  ({key_name(self.cfg.ptt_key)})", menu)
        hotkey.triggered.connect(self._change_hotkey)

        startup = QAction("Start with Windows", menu, checkable=True)
        startup.setChecked(is_run_on_startup())
        startup.toggled.connect(self._set_run_on_startup)

        reset = QAction("Reset position", menu)
        reset.triggered.connect(self.window.reset_position)

        quit_action = QAction("Quit Voice Flow", menu)
        quit_action.triggered.connect(self.shutdown)

        menu.addActions([show, copy_last])
        menu.addSeparator()
        menu.addAction(send)
        menu.addAction(hotkey)
        if sys.platform.startswith("win"):
            menu.addAction(startup)
        menu.addSeparator()
        menu.addAction(reset)
        menu.addSeparator()
        menu.addAction(quit_action)

    def _show_context_menu(self, global_pos):
        menu = QMenu()
        self._populate_menu(menu)
        menu.exec(global_pos)

    def _toggle_record(self):
        """Click on the pill starts or stops a recording (mouse-only path)."""
        if self.engine.awaiting_retry:            # a failed download → click retries it
            self.window.set_state("downloading", "Retrying…")
            self.engine.retry()
            return
        if not self.engine.ready:
            return
        self.hotkeys.clear_latch()
        if self.engine.recording:
            self.engine.stop()
        else:
            self.engine.start()

    def _set_auto_send(self, value: bool):
        self.cfg.auto_send = value
        QSettings("VoiceFlow", "settings").setValue("auto_send", value)

    def _set_run_on_startup(self, value: bool):
        set_run_on_startup(value)

    def _change_hotkey(self):
        self.window.set_state("idle", "Press the new hotkey…  (Esc to keep current)")
        # capture runs on the listener thread → hop back to the UI via the signal
        self.hotkeys.capture_next(self.signals.hotkey_captured.emit)

    def _on_hotkey_captured(self, key):
        if key is not None:
            self.cfg.ptt_key = key
            QSettings("VoiceFlow", "settings").setValue("ptt_key", key_to_str(key))
            self.window.set_state("idle", f"Hotkey set to {key_name(key)}")
        else:
            self.window.set_state("idle", "Kept the current hotkey")
        self.tray_menu.clear()
        self._populate_menu(self.tray_menu)

    # ---- state handlers ----------------------------------------------------
    def _on_ready(self):
        name = key_name(self.cfg.ptt_key)
        self.window.set_state("idle", f"Hold {name} to speak  ·  Esc Esc to quit")
        s = QSettings("VoiceFlow", "settings")
        if not s.value("onboarded"):
            self.window.flash_message(f"Hold {name} and speak — or click me")
            s.setValue("onboarded", True)

    def _on_downloading(self, frac: float):
        if frac < 0:
            self.window.set_state("downloading", "Downloading speech model…")
        else:
            self.window.set_state("downloading", f"Downloading speech model… {int(frac * 100)}%")

    def _on_download_failed(self, message: str):
        self.window.set_state("error", f"Download failed — click to retry  ({message})")

    def _on_listening(self, latched: bool):
        if latched:
            self.window.set_state("latched", "Tap Right Ctrl to finish — or just stop talking")
        else:
            self.window.set_state("listening", "Release Right Ctrl when you're done")

    def _on_working(self):
        self.hotkeys.clear_latch()
        self.window.set_state("working", "Turning speech into text…", timer="")

    def _on_done(self, text: str, seconds: float):
        self.last_text = text
        words = len(text.split())
        self.window.set_state("done", text, timer=f"{words}w · {seconds:.1f}s")
        QTimer.singleShot(1800, lambda: self._on_ready() if self.window.state == "done" else None)

    def _on_cancelled(self):
        self.hotkeys.clear_latch()
        self.window.set_state("idle", "Cancelled — nothing was pasted")

    def _on_error(self, message: str):
        self.hotkeys.clear_latch()
        self.window.set_state("error", message, timer="")
        QTimer.singleShot(2600, lambda: self._on_ready() if self.window.state == "error" else None)

    def _cancel(self):
        self.hotkeys.clear_latch()
        self.engine.stop(cancel=True)

    def _guard(self):
        """Stops runaway recordings and finishes hands-free mode after silence."""
        if not self.engine.recording:
            return
        if self.engine.elapsed() > self.cfg.max_record_seconds:
            self.hotkeys.clear_latch()
            self.engine.stop()
        elif self.engine.latched and self.engine.silent_for() > self.cfg.silence_stop_seconds:
            self.hotkeys.clear_latch()
            self.engine.stop()

    def shutdown(self):
        """Tear everything down, then leave. Called from the tray, Esc Esc, or a signal."""
        print("[voice-flow] shutting down")
        try:
            self.guard.stop()
        except Exception:
            pass
        try:
            self.tray.hide()
        except Exception:
            pass
        try:
            self.hotkeys.stop()
        except Exception:
            pass
        try:
            self.engine.stop(cancel=True)
        except Exception:
            pass
        self.qt.quit()

    def run(self):
        code = self.qt.exec()
        self.shutdown()
        # PortAudio and the key listener can both refuse to unwind cleanly.
        # Flush stdout, then leave without waiting for them.
        sys.stdout.flush()
        os._exit(code)


if __name__ == "__main__":
    App(CFG).run()