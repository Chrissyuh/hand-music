import os
import time
import json
import threading
import urllib.request
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple, Set

import cv2
import numpy as np
import mediapipe as mp
import mido

# ============================================================
# Hand Music (MIDI) - Main hand = zones/spread; Control hand = 1 analog control
# - Control hand: thumb↔pinky distance = REVERB (PAD) + ARP tempo (ARP)
# - PAD/ARP switch stays on MAIN hand pinch (thumb-index) when NOT in fist
# - Top UI (bars + info) is GREEN
# ============================================================

# ----------------------------
# MIDI config
# ----------------------------
MIDI_OUT_NAME_HINT = "Microsoft GS Wavetable Synth"  # or leave empty to pick first port
CH_BASE = 0
CH_MELODY = 1

SEND_PROGRAM_CHANGE = True
PROG_BASE = 89     # warm pad-ish (General MIDI: 89 = Pad 2 "Warm")
PROG_MELODY = 90   # synth pad/voice-ish (90 = Pad 3 "Polysynth") or tweak

# CCs
CC_EXPR = 11
CC_MOD = 1
CC_REVERB = 91
CC_CHORUS = 93
CC_PAN = 10

# ----------------------------
# Performance + smoothing
# ----------------------------
TARGET_FPS = 30
FRAME_DOWNSCALE = 0.75  # <1.0 to speed up; 1.0 full res
EMA_ALPHA_Y = 0.25
EMA_ALPHA_PALM = 0.25
EMA_ALPHA_SPREAD = 0.25
EMA_ALPHA_FIST = 0.25
EMA_ALPHA_CC1 = 0.20
EMA_ALPHA_CTRL_PINKY = 0.20

# ----------------------------
# Zones + mapping
# ----------------------------
NUM_ZONES = 7
ZONE_MARGIN_X = 0.06
ZONE_MARGIN_Y = 0.08

# Distance mapping defaults (MAIN hand)
DIST_FAR_DEFAULT = 0.35   # farther = lower pitch zone
DIST_NEAR_DEFAULT = 0.10  # nearer = higher pitch zone

# Spread mapping defaults (thumb↔pinky)
SPREAD_MIN_DEFAULT = 0.45
SPREAD_MAX_DEFAULT = 0.95

# Fist mapping (raw fistness -> logic)
# Raw fistness tends to be ~0.47 .. 0.86 for you; expand slightly beyond.
FIST_RAW_MIN = 0.44
FIST_RAW_MAX = 0.89
FIST_ON_THRESH = 0.78
FIST_OFF_THRESH = 0.72

# Switching modes (MAIN pinch)
PINCH_ON = 0.24
PINCH_OFF = 0.30
PINCH_HOLD_MS = 220

# ----------------------------
# Slur behavior
# ----------------------------
SLUR_OVERLAP_MS = 45  # overlap note-offs by this many ms so changes feel like slurs

# ----------------------------
# Arp settings
# ----------------------------
# Default step; in ARP mode this can be overridden by control-hand thumb↔pinky distance.
ARP_STEP_DEFAULT_MS = 220
ARP_STEP_MIN_MS = 110
ARP_STEP_MAX_MS = 520

# Note length is derived from the step (a little legato) but clamped.
ARP_NOTE_FRACTION = 0.72
ARP_NOTE_MIN_MS = 70
ARP_NOTE_MAX_MS = 240

# ----------------------------
# Visuals
# ----------------------------
WIN_NAME = "Hand Music MIDI"
TOP_UI_COLOR = (0, 255, 0)  # green
TOP_UI_BG = (0, 0, 0)

ORANGE = (0, 128, 255)
PINK = (255, 0, 255)
WHITE = (255, 255, 255)

DRAW_SKELETON = True
SHOW_HAND_BOX = True
SHOW_CURSOR = True

UI_FONT = cv2.FONT_HERSHEY_SIMPLEX
UI_SCALE = 0.55
UI_THICKNESS = 1

BAR_THICKNESS = 1

# ----------------------------
# Control-hand thumb↔pinky mapping
# (same direction as spread: smaller=0, larger=1)
# ----------------------------
CTRL_PINKY_MIN = 0.38
CTRL_PINKY_MAX = 0.95

# ----------------------------
# Chords (original set)
# ----------------------------
@dataclass
class Chord:
    name: str
    notes: List[int]

CHORDS: List[Chord] = [
    Chord("D(add9/6)",      [50, 57, 62, 64, 66]),
    Chord("Em11(no3)",      [52, 59, 62, 69, 71]),
    Chord("F#m11(no3)",     [54, 61, 64, 69, 71]),
    Chord("G(add9/6)",      [55, 62, 67, 69, 71]),
    Chord("Asus2(add4)",    [57, 64, 69, 71, 74]),
    # Top-2 reverted (original set), order fixed:
    Chord("Dmaj9",          [50, 57, 61, 64, 66]),  # this used to sound "too low" in zone 6 without octave lift
    Chord("Bm11",           [59, 66, 69, 74, 76]),
]

# Lift higher zones up so zone 6 won't "drop" and zone 7 stays highest.
OCTAVE_LIFT_BY_ZONE = [0, 0, 0, 0, 0, 12, 12]

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def ensure_model_file(model_path: str) -> str:
    """
    Ensure mediapipe hand_landmarker.task exists.
    If missing, download from official repo URL.
    """
    if os.path.exists(model_path):
        return model_path
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    print("Downloading model:", url)
    urllib.request.urlretrieve(url, model_path)
    return model_path


def clamp01(x: float) -> float:
    return 0.0 if x <= 0.0 else (1.0 if x >= 1.0 else float(x))


def norm_to_pixel(lm, w: int, h: int) -> Tuple[int, int]:
    return int(lm.x * w), int(lm.y * h)


def draw_text_plain(img, text, org, color=WHITE, scale=0.5, thickness=1):
    # small shadow for readability
    x, y = org
    cv2.putText(img, text, (x + 1, y + 1), UI_FONT, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), UI_FONT, scale, color, thickness, cv2.LINE_AA)


def draw_bar(img, x, y, w, h, value01: float, label: str):
    value01 = clamp01(value01)
    # border
    cv2.rectangle(img, (x, y), (x + w, y + h), TOP_UI_COLOR, BAR_THICKNESS)
    # fill
    fill_w = int(w * value01)
    if fill_w > 0:
        cv2.rectangle(img, (x + 1, y + 1), (x + fill_w, y + h - 1), TOP_UI_COLOR, -1)
    draw_text_plain(img, label, (x + w + 8, y + h - 2), color=TOP_UI_COLOR, scale=0.50, thickness=1)


def palm_center_norm(lms) -> Tuple[float, float]:
    # use wrist(0) + middle mcp(9)
    cx = (lms[0].x + lms[9].x) * 0.5
    cy = (lms[0].y + lms[9].y) * 0.5
    return cx, cy


def palm_scale_norm(lms) -> float:
    # distance between wrist(0) and middle mcp(9) is a stable "palm scale"
    dx = lms[9].x - lms[0].x
    dy = lms[9].y - lms[0].y
    d = float(np.sqrt(dx * dx + dy * dy))
    return max(1e-6, d)


def spread_thumb_pinky_norm(lms, palm_scale: float) -> float:
    # distance between thumb tip(4) and pinky tip(20), normalized
    dx = lms[20].x - lms[4].x
    dy = lms[20].y - lms[4].y
    d = float(np.sqrt(dx * dx + dy * dy)) / palm_scale
    return d


def pinch_thumb_index_norm(lms, palm_scale: float) -> float:
    dx = lms[8].x - lms[4].x
    dy = lms[8].y - lms[4].y
    d = float(np.sqrt(dx * dx + dy * dy)) / palm_scale
    return d


def pinch_thumb_pinky_norm(lms, palm_scale: float) -> float:
    dx = lms[20].x - lms[4].x
    dy = lms[20].y - lms[4].y
    d = float(np.sqrt(dx * dx + dy * dy)) / palm_scale
    return d


def fistness_raw(lms, palm_scale: float) -> float:
    """
    Raw fistness heuristic:
    Average distance of finger tips to their MCPs (tip closer to MCP => more fist).
    """
    pairs = [(8, 5), (12, 9), (16, 13), (20, 17)]
    ds = []
    for tip, mcp in pairs:
        dx = lms[tip].x - lms[mcp].x
        dy = lms[tip].y - lms[mcp].y
        ds.append(float(np.sqrt(dx * dx + dy * dy)) / palm_scale)
    # smaller distance => more fist; invert-ish
    avg = float(np.mean(ds))
    raw = 1.0 - clamp01((avg - 0.20) / 0.45)
    return clamp01(raw)


def fistness_mapped_for_logic(raw: float) -> float:
    # remap using your observed range (with small buffer)
    if FIST_RAW_MAX <= FIST_RAW_MIN:
        return raw
    return clamp01((raw - FIST_RAW_MIN) / (FIST_RAW_MAX - FIST_RAW_MIN))


def zone_from_x(cx_norm: float, num_zones: int) -> int:
    # clamp margins, then map
    x = float(np.clip(cx_norm, ZONE_MARGIN_X, 1.0 - ZONE_MARGIN_X))
    usable = 1.0 - 2.0 * ZONE_MARGIN_X
    t = (x - ZONE_MARGIN_X) / usable
    z = int(np.floor(t * num_zones))
    return int(np.clip(z, 0, num_zones - 1))


def transpose_from_dist(dist_norm: float, near_val: float, far_val: float) -> int:
    """
    dist_norm: normalized palm distance estimate (smaller = closer)
    near_val -> highest
    far_val  -> lowest
    output: semitone transpose in range [-12..+12]
    """
    # map dist_norm into 0..1 (near->1, far->0)
    if far_val <= near_val:
        t = 0.5
    else:
        t = 1.0 - clamp01((dist_norm - near_val) / (far_val - near_val))
    # semitones
    semi = int(round((t - 0.5) * 24.0))
    return int(np.clip(semi, -12, 12))


def chord_velocity_comp(notes: List[int]) -> int:
    """
    Mild velocity adjustment so super-high doesn't stab and super-low doesn't vanish.
    """
    if not notes:
        return 0
    avg = float(np.mean(notes))
    target = 62.0
    boost = int(round((target - avg) * 0.55))
    return int(np.clip(boost, -12, 14))


def draw_hand_skeleton(img, lms, w, h, color=(0, 255, 0)):
    # connections (MediaPipe hands)
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17),
    ]
    for a, b in connections:
        pa = norm_to_pixel(lms[a], w, h)
        pb = norm_to_pixel(lms[b], w, h)
        cv2.line(img, pa, pb, color, 1)


# ------------------------------------------------------------
# MIDI util
# ------------------------------------------------------------
def pick_midi_port() -> mido.ports.BaseOutput:
    names = mido.get_output_names()
    if not names:
        raise RuntimeError("No MIDI output ports found. Install a synth or enable Microsoft GS Wavetable Synth.")
    if MIDI_OUT_NAME_HINT:
        for n in names:
            if MIDI_OUT_NAME_HINT.lower() in n.lower():
                return mido.open_output(n)
    return mido.open_output(names[0])


def midi_msg(outport, msg: mido.Message):
    try:
        outport.send(msg)
    except Exception:
        pass


# ------------------------------------------------------------
# Frame grabber thread (reduces latency)
# ------------------------------------------------------------
class FrameGrabber:
    def __init__(self, cap: cv2.VideoCapture):
        self.cap = cap
        self.lock = threading.Lock()
        self.frame = None
        self.stopped = False
        self.t = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.t.start()
        return self

    def _run(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.005)
                continue
            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def stop(self):
        self.stopped = True


# ------------------------------------------------------------
# Recorder (optional) - writes metrics to JSONL
# ------------------------------------------------------------
class Recorder:
    def __init__(self, path="recording.jsonl"):
        self.path = path
        self.recording = False
        self.f = None

    def toggle(self):
        if self.recording:
            self.stop()
        else:
            self.start()

    def start(self):
        self.f = open(self.path, "a", encoding="utf-8")
        self.recording = True

    def stop(self):
        if self.f:
            self.f.close()
        self.f = None
        self.recording = False

    def log(self, d: Dict[str, Any]):
        if not self.recording or self.f is None:
            return
        self.f.write(json.dumps(d) + "\n")
        self.f.flush()


# ------------------------------------------------------------
# Sound presets (tweakable)
# ------------------------------------------------------------
@dataclass
class SoundPreset:
    name: str
    base_velocity: int
    base_reverb: int
    base_chorus: int

SOUND_PRESETS: List[SoundPreset] = [
    SoundPreset("Cinematic Pad", base_velocity=78, base_reverb=55, base_chorus=24),
    SoundPreset("Ethereal Soft", base_velocity=70, base_reverb=72, base_chorus=18),
    SoundPreset("Dark Glass",    base_velocity=84, base_reverb=60, base_chorus=28),
]

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    # MIDI
    outport = pick_midi_port()
    print("Using MIDI out:", outport.name)

    def cc(ch: int, ccnum: int, val: int):
        midi_msg(outport, mido.Message("control_change", channel=ch, control=ccnum, value=int(val)))

    def program(ch: int, prog: int):
        midi_msg(outport, mido.Message("program_change", channel=ch, program=int(prog)))

    def note_on(ch: int, note: int, vel: int):
        midi_msg(outport, mido.Message("note_on", channel=ch, note=int(note), velocity=int(vel)))

    def note_off(ch: int, note: int):
        midi_msg(outport, mido.Message("note_off", channel=ch, note=int(note), velocity=0))

    def pitchbend(ch: int, val: int):
        # val in [-8192..8191]
        midi_msg(outport, mido.Message("pitchwheel", channel=ch, pitch=int(val)))

    # Init programs and CC
    preset_idx = 0
    preset = SOUND_PRESETS[preset_idx]
    if SEND_PROGRAM_CHANGE:
        program(CH_BASE, PROG_BASE)
        program(CH_MELODY, PROG_MELODY)

    # Base FX
    cc(CH_BASE, CC_REVERB, preset.base_reverb)
    cc(CH_MELODY, CC_REVERB, preset.base_reverb)
    cc(CH_BASE, CC_CHORUS, preset.base_chorus)
    cc(CH_MELODY, CC_CHORUS, preset.base_chorus)
    cc(CH_BASE, CC_PAN, 64)
    cc(CH_MELODY, CC_PAN, 64)

    # open camera
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    grabber = FrameGrabber(cap).start()

    # mediapipe tasks hand landmarker
    model_path = ensure_model_file(os.path.join(os.path.dirname(__file__), "models", "hand_landmarker.task"))
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    landmarker = HandLandmarker.create_from_options(options)

    recorder = Recorder()

    print("\nControl-hand (2nd hand):")
    print("  Thumb↔pinky distance controls:")
    print("    - PAD mode: REVERB amount (analog)")
    print("    - ARP mode: ARP tempo (open = faster)")
    print("\nMain hand:")
    print("  Thumb-index pinch toggles PAD/ARP when NOT in fist\n")

    # Smoothed metrics (MAIN hand)
    ema_y: Optional[float] = None
    ema_palm: Optional[float] = None
    ema_spread: Optional[float] = None
    ema_fist_logic: Optional[float] = None
    ema_cc1: Optional[float] = None

    # Control hand analog EMA
    ema_ctrl_pinky: Optional[float] = None

    # Calibration (MAIN)
    dist_far = DIST_FAR_DEFAULT
    dist_near = DIST_NEAR_DEFAULT
    spread_min = SPREAD_MIN_DEFAULT
    spread_max = SPREAD_MAX_DEFAULT

    # Last seen values (for sanity display)
    last_spread_seen = 0.0
    last_palm_seen = 0.0
    last_fist_raw = 0.0

    # MIDI state (PAD)
    held_base: Set[int] = set()
    held_mel: Set[int] = set()
    desired_base: Set[int] = set()
    desired_mel: Set[int] = set()
    pending_off_pad: List[Tuple[int, int, int]] = []  # (time_ms, ch, note)

    # MIDI state (ARP)
    arp_notes: List[int] = []
    arp_index = 0
    arp_next_ms = 0
    pending_off_arp: List[Tuple[int, int, int]] = []  # (time_ms, ch, note)

    # Mode switch state
    mode_arp: bool = False
    pinch_active: bool = False
    pinch_started_ms: Optional[int] = None
    pinch_consumed: bool = False

    # Control-hand (analog only)
    last_cc91 = preset.base_reverb
    last_cc93 = preset.base_chorus

    # main-hand selection preference when 2 hands
    main_pref_right = True
    last_main_center: Optional[Tuple[float, float]] = None

    # Expression cap (no lofi now)
    expr_cap = MAX_EXPR = 127
    MIN_EXPR = 30

    # info/debug
    last_result_data: Dict[str, Any] = {}

    def stop_all_now() -> None:
        nonlocal held_base, held_mel, pending_off_pad, pending_off_arp, arp_notes
        for n in list(held_base):
            note_off(CH_BASE, n)
        for n in list(held_mel):
            note_off(CH_MELODY, n)
        held_base.clear()
        held_mel.clear()
        desired_base.clear()
        desired_mel.clear()
        pending_off_pad.clear()
        pending_off_arp.clear()
        arp_notes = []

    # Window
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)

    prev_time = time.time()
    frame_id = 0

    while True:
        frame = grabber.read()
        if frame is None:
            time.sleep(0.001)
            continue

        # Mirror camera
        frame = cv2.flip(frame, 1)

        # Resize for speed
        if FRAME_DOWNSCALE != 1.0:
            frame = cv2.resize(frame, None, fx=FRAME_DOWNSCALE, fy=FRAME_DOWNSCALE, interpolation=cv2.INTER_AREA)

        h, w = frame.shape[:2]
        frame_bgr = frame  # already BGR
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        now_ms = int(time.time() * 1000)

        # Mediapipe expects mp.Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # detect
        result = landmarker.detect_for_video(mp_image, now_ms)
        hands = result.hand_landmarks if result and result.hand_landmarks else []

        # clear desired each frame
        desired_base.clear()
        desired_mel.clear()

        # Handle pending offs
        if pending_off_pad:
            pending_off_pad.sort(key=lambda x: x[0])
            while pending_off_pad and pending_off_pad[0][0] <= now_ms:
                _, ch, n = pending_off_pad.pop(0)
                # only turn off if not desired anymore
                if ch == CH_BASE and n not in desired_base and n in held_base:
                    note_off(ch, n)
                    held_base.discard(n)
                if ch == CH_MELODY and n not in desired_mel and n in held_mel:
                    note_off(ch, n)
                    held_mel.discard(n)

        if pending_off_arp:
            pending_off_arp.sort(key=lambda x: x[0])
            while pending_off_arp and pending_off_arp[0][0] <= now_ms:
                _, ch, n = pending_off_arp.pop(0)
                note_off(ch, n)

        # If no hand, decay: stop everything if nothing seen recently
        if not hands:
            # If in PAD holding notes, release them gracefully (slur overlap not needed)
            if held_base or held_mel:
                stop_all_now()
            # draw help text
            draw_text_plain(frame_bgr, "Show hand to play", (12, 26), color=TOP_UI_COLOR, scale=0.70, thickness=1)
            cv2.imshow(WIN_NAME, frame_bgr)
            if cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
            continue

        # choose main + control hand
        main_lms = None
        ctrl_lms = None
        if len(hands) == 1:
            main_lms = hands[0]
            ctrl_lms = None
        else:
            centers = [palm_center_norm(h) for h in hands[:2]]
            # decide rightmost as main by default; keep preference stable
            if last_main_center is not None:
                d0 = (centers[0][0] - last_main_center[0]) ** 2 + (centers[0][1] - last_main_center[1]) ** 2
                d1 = (centers[1][0] - last_main_center[0]) ** 2 + (centers[1][1] - last_main_center[1]) ** 2
                main_idx = 0 if d0 <= d1 else 1
                ctrl_idx = 1 - main_idx
            else:
                # default: rightmost is main
                main_idx = 0 if centers[0][0] >= centers[1][0] else 1
                ctrl_idx = 1 - main_idx

            main_lms = hands[main_idx]
            last_main_center = palm_center_norm(main_lms)

            ctrl_lms = hands[ctrl_idx] if ctrl_idx is not None else None
            # ------------------------------------------------------------
            # Control-hand analog (thumb↔pinky): drives REVERB (PAD) and ARP tempo (ARP)
            # ------------------------------------------------------------
            ctrl_pinky01: Optional[float] = None
            ctrl_pinky_dist: Optional[float] = None

            if ctrl_lms is not None:
                cps = palm_scale_norm(ctrl_lms)
                c_pinky = pinch_thumb_pinky_norm(ctrl_lms, cps)  # thumb(4) ↔ pinky tip(20), normalized by palm_scale
                ctrl_pinky_dist = float(c_pinky)

                # Map to 0..1 (same direction as spread mapping: smaller=0, larger=1)
                if CTRL_PINKY_MAX <= CTRL_PINKY_MIN:
                    ctrl_pinky01 = 0.0
                else:
                    ctrl_pinky01 = clamp01((ctrl_pinky_dist - CTRL_PINKY_MIN) / (CTRL_PINKY_MAX - CTRL_PINKY_MIN))

                # EMA smooth
                ema_ctrl_pinky = ctrl_pinky01 if ema_ctrl_pinky is None else (
                    EMA_ALPHA_CTRL_PINKY * ctrl_pinky01 + (1.0 - EMA_ALPHA_CTRL_PINKY) * float(ema_ctrl_pinky)
                )
                ctrl_pinky01 = float(ema_ctrl_pinky)

                # Draw the one control line (thumb -> pinky), colored to stand out
                c_thumb = norm_to_pixel(ctrl_lms[4], w, h)
                c20p = norm_to_pixel(ctrl_lms[20], w, h)
                cv2.line(frame_bgr, c_thumb, c20p, (0, 200, 255), 3)
                lx = int(np.clip(c20p[0], 6, w - 230))
                ly = int(np.clip(c20p[1] - 10, 18, h - 6))
                draw_text_plain(frame_bgr, f"ctrl(4-20)={ctrl_pinky_dist:.3f}", (lx, ly), color=(0, 200, 255))

            # Apply reverb (stronger range). If control hand disappears, revert to preset default.
            if ctrl_pinky01 is None:
                target_reverb = preset.base_reverb
            else:
                target_reverb = int(round(15 + ctrl_pinky01 * 112))  # 15..127
            target_reverb = max(0, min(127, target_reverb))

            if abs(target_reverb - last_cc91) >= 2:
                for ch in (CH_BASE, CH_MELODY):
                    cc(ch, CC_REVERB, target_reverb)
                last_cc91 = target_reverb

            # ARP timing: use control hand in ARP mode; otherwise use default
            if mode_arp and (ctrl_pinky01 is not None):
                arp_step_ms_curr = int(round(ARP_STEP_MAX_MS - ctrl_pinky01 * (ARP_STEP_MAX_MS - ARP_STEP_MIN_MS)))
                arp_step_ms_curr = int(np.clip(arp_step_ms_curr, ARP_STEP_MIN_MS, ARP_STEP_MAX_MS))
            else:
                arp_step_ms_curr = ARP_STEP_DEFAULT_MS

            arp_note_ms_curr = int(round(arp_step_ms_curr * ARP_NOTE_FRACTION))
            arp_note_ms_curr = int(np.clip(arp_note_ms_curr, ARP_NOTE_MIN_MS, ARP_NOTE_MAX_MS))

        # Draw skeletons
        if DRAW_SKELETON and main_lms is not None:
            draw_hand_skeleton(frame_bgr, main_lms, w, h, color=(0, 255, 0))
            if ctrl_lms is not None:
                draw_hand_skeleton(frame_bgr, ctrl_lms, w, h, color=(255, 0, 255))

        # MAIN bbox (orange) + cursor
        if main_lms is not None and SHOW_HAND_BOX:
            xs = [lm.x for lm in main_lms]
            ys = [lm.y for lm in main_lms]
            min_x, max_x = int(min(xs) * w), int(max(xs) * w)
            min_y, max_y = int(min(ys) * h), int(max(ys) * h)
            cv2.rectangle(frame_bgr, (min_x, min_y), (max_x, max_y), ORANGE, 2)

            if SHOW_CURSOR:
                # cursor from index tip (8)
                p8 = norm_to_pixel(main_lms[8], w, h)
                cv2.circle(frame_bgr, p8, 6, PINK, -1)

            # bbox metrics text (right/bottom anchored)
            bbox_w = max_x - min_x
            bbox_h = max_y - min_y
            anchor_x = max_x + 6
            anchor_y = max_y + 0
            anchor_x = int(np.clip(anchor_x, 6, w - 240))
            anchor_y = int(np.clip(anchor_y, 26, h - 6))
            draw_text_plain(frame_bgr, f"bbox L={min_x} T={min_y} W={bbox_w} H={bbox_h}", (anchor_x, anchor_y),
                            color=ORANGE, scale=0.50, thickness=1)

        # MAIN processing
        if main_lms is not None:
            # Center
            cx, cy = palm_center_norm(main_lms)
            ps = palm_scale_norm(main_lms)
            last_palm_seen = ps

            spread_norm = spread_thumb_pinky_norm(main_lms, ps)
            last_spread_seen = spread_norm

            pinch_norm = pinch_thumb_index_norm(main_lms, ps)
            # Visualize MAIN hand pinch (thumb↔index) used for PAD/ARP switching
            p4_main = norm_to_pixel(main_lms[4], w, h)
            p8_main = norm_to_pixel(main_lms[8], w, h)
            cv2.line(frame_bgr, p4_main, p8_main, (0, 255, 255), 2)
            # simple label near the midpoint
            mx = (p4_main[0] + p8_main[0]) // 2
            my = (p4_main[1] + p8_main[1]) // 2
            mx = int(np.clip(mx, 6, w - 140))
            my = int(np.clip(my - 8, 18, h - 6))
            draw_text_plain(frame_bgr, f"pinch(4-8)={pinch_norm:.3f}", (mx, my), color=(0, 255, 255))

            f_raw = fistness_raw(main_lms, ps)
            last_fist_raw = f_raw
            f_logic = fistness_mapped_for_logic(f_raw)

            # EMA
            ema_y = cy if ema_y is None else (EMA_ALPHA_Y * cy + (1.0 - EMA_ALPHA_Y) * ema_y)
            ema_palm = ps if ema_palm is None else (EMA_ALPHA_PALM * ps + (1.0 - EMA_ALPHA_PALM) * ema_palm)
            ema_spread = spread_norm if ema_spread is None else (EMA_ALPHA_SPREAD * spread_norm + (1.0 - EMA_ALPHA_SPREAD) * ema_spread)
            ema_fist_logic = f_logic if ema_fist_logic is None else (EMA_ALPHA_FIST * f_logic + (1.0 - EMA_ALPHA_FIST) * ema_fist_logic)

            # Update calibrations slowly (optional, conservative)
            # We'll nudge min/max toward observed values only a tiny bit.
            dist_far = 0.995 * dist_far + 0.005 * max(dist_far, float(ema_palm))
            dist_near = 0.995 * dist_near + 0.005 * min(dist_near, float(ema_palm))
            spread_min = 0.995 * spread_min + 0.005 * min(spread_min, float(ema_spread))
            spread_max = 0.995 * spread_max + 0.005 * max(spread_max, float(ema_spread))

            # Determine fist state with hysteresis
            fist_is_on = bool(ema_fist_logic >= FIST_ON_THRESH)
            fist_is_off = bool(ema_fist_logic <= FIST_OFF_THRESH)
            if not hasattr(main, "_fist_state"):
                main._fist_state = False
            if main._fist_state:
                if fist_is_off:
                    main._fist_state = False
            else:
                if fist_is_on:
                    main._fist_state = True
            fist_is_on = bool(main._fist_state)

            # Determine pinch state for mode toggle (only if NOT fist)
            if not fist_is_on:
                if pinch_active:
                    if pinch_norm >= PINCH_OFF:
                        pinch_active = False
                        pinch_started_ms = None
                        pinch_consumed = False
                else:
                    if pinch_norm <= PINCH_ON:
                        pinch_active = True
                        pinch_started_ms = now_ms
                        pinch_consumed = False

                if pinch_active and (pinch_started_ms is not None) and (not pinch_consumed):
                    if now_ms - pinch_started_ms >= PINCH_HOLD_MS:
                        mode_arp = not mode_arp
                        pinch_consumed = True
                        # when switching modes, release any held PAD notes
                        stop_all_now()
                        arp_notes = []
                        arp_index = 0
                        arp_next_ms = now_ms
            else:
                pinch_active = False
                pinch_started_ms = None
                pinch_consumed = False

            # Determine zone from x
            active_zone = zone_from_x(cx, NUM_ZONES)

            # Determine transpose from palm scale (distance-like)
            transpose = transpose_from_dist(float(ema_palm), dist_near, dist_far)

            # Determine chord
            chord = CHORDS[active_zone % len(CHORDS)]
            lift = OCTAVE_LIFT_BY_ZONE[active_zone]
            chord_notes = sorted(chord.notes)
            chord_notes = [min(127, max(0, n + transpose + lift)) for n in chord_notes]

            # Soften the very top chord (zone 7/7) so it doesn't get painfully high.
            # We keep the zone mapping as-is, but "drop" the highest voice if it crosses a harsh register.
            if active_zone == (NUM_ZONES - 1) and chord_notes:
                if chord_notes[-1] >= 85:
                    chord_notes[-1] = max(0, chord_notes[-1] - 12)
                    chord_notes.sort()

            # Spread -> expression
            if spread_max <= spread_min:
                spread01 = 0.5
            else:
                spread01 = clamp01((float(ema_spread) - spread_min) / (spread_max - spread_min))

            expr = int(round(MIN_EXPR + spread01 * (expr_cap - MIN_EXPR)))
            expr = max(0, min(127, expr))
            # smooth expr send
            last_expr_sent = getattr(main, "_last_expr_sent", None)
            if last_expr_sent is None:
                last_expr_sent = expr
            else:
                # small smoothing
                last_expr_sent = int(round(0.65 * last_expr_sent + 0.35 * expr))
            main._last_expr_sent = last_expr_sent

            cc(CH_BASE, CC_EXPR, last_expr_sent)
            cc(CH_MELODY, CC_EXPR, last_expr_sent)

            # Vib -> Mod wheel
            # Use vertical position (higher hand -> more vib) but subtle
            vib01 = clamp01(1.0 - float(ema_y))
            mod_val = int(round(vib01 * 90))
            if getattr(main, "_last_mod", None) is None:
                main._last_mod = mod_val
            else:
                main._last_mod = int(round(EMA_ALPHA_CC1 * mod_val + (1.0 - EMA_ALPHA_CC1) * main._last_mod))
            mod_val = int(np.clip(main._last_mod, 0, 127))

            cc(CH_BASE, CC_MOD, mod_val)
            cc(CH_MELODY, CC_MOD, mod_val)

            # Prepare PAD notes: bed + top
            top_note = chord_notes[-1]
            bed_notes = chord_notes[:-1]

            # Dynamic velocity base
            base_vel = preset.base_velocity
            base_vel = max(1, min(127, base_vel))
            vel = int(np.clip(base_vel + chord_velocity_comp(chord_notes), 1, 127))

            # Handle fist: fade out / stop
            releasing_fist = fist_is_on

            if not mode_arp:
                # PAD mode: hold chord with slurred changes
                if not releasing_fist:
                    desired_base.update(bed_notes)
                    desired_mel.add(top_note)

                # Apply slur logic: turn ON any missing desired notes immediately,
                # turn OFF old notes after a small overlap.
                for n in sorted(list(desired_base - held_base)):
                    note_on(CH_BASE, n, vel)
                    held_base.add(n)
                for n in sorted(list(desired_mel - held_mel)):
                    note_on(CH_MELODY, n, vel)
                    held_mel.add(n)

                for n in sorted(list(held_base - desired_base)):
                    pending_off_pad.append((now_ms + SLUR_OVERLAP_MS, CH_BASE, n))
                for n in sorted(list(held_mel - desired_mel)):
                    pending_off_pad.append((now_ms + SLUR_OVERLAP_MS, CH_MELODY, n))

            # ARP mode: arpeggiate chord
            if mode_arp:
                # update arp_notes when chord changes or empty
                if not arp_notes or getattr(main, "_last_zone", None) != active_zone or getattr(main, "_last_transpose", None) != transpose:
                    notes_up = chord_notes[:]
                    notes_down = chord_notes[-2:0:-1] if len(notes_up) > 2 else []
                    arp_notes = notes_up + notes_down
                    arp_index = 0
                    arp_next_ms = now_ms
                    main._last_zone = active_zone
                    main._last_transpose = transpose

                # ARP engine (full chord)
                if (not releasing_fist) and (not fist_is_on):
                    if arp_notes and now_ms >= arp_next_ms:
                        n = arp_notes[arp_index % len(arp_notes)]
                        arp_index += 1
                        arp_next_ms = now_ms + arp_step_ms_curr

                        ch = CH_MELODY
                        # Make ARP changes obvious: run all arp notes through the brighter melody channel.
                        vel_arp = vel if n >= (top_note - 12) else max(1, vel - 10)
                        note_on(ch, n, vel_arp)
                        pending_off_arp.append((now_ms + arp_note_ms_curr, ch, n))

            # Spread line (thumb ↔ pinky) + label pinned to pinky
            p4 = norm_to_pixel(main_lms[4], w, h)
            p20 = norm_to_pixel(main_lms[20], w, h)
            cv2.line(frame_bgr, p4, p20, (255, 255, 255), 2)
            # label near pinky
            lx = int(np.clip(p20[0], 6, w - 220))
            ly = int(np.clip(p20[1] - 12, 18, h - 6))
            draw_text_plain(frame_bgr, f"spread(4-20)={float(ema_spread):.3f}", (lx, ly), color=WHITE, scale=0.50, thickness=1)

            # ----------------------------
            # (Control-hand processed above)
            #
            # Bars (TOP UI) + info (TOP UI) — GREEN
            # ----------------------------
            panel_x, panel_y = 10, 10
            bar_w, bar_h, gap = 140, 10, 14

            draw_bar(frame_bgr, panel_x, panel_y + 0 * gap, bar_w, bar_h, spread01, "spread")
            draw_bar(frame_bgr, panel_x, panel_y + 1 * gap, bar_w, bar_h, float(ema_fist_logic), "fist")
            expr01 = clamp01((float(last_expr_sent) - MIN_EXPR) / max(1.0, float(expr_cap - MIN_EXPR))) if last_expr_sent is not None else 0.0
            draw_bar(frame_bgr, panel_x, panel_y + 2 * gap, bar_w, bar_h, expr01, "expr")
            draw_bar(frame_bgr, panel_x, panel_y + 3 * gap, bar_w, bar_h, vib01, "vib")
            draw_bar(frame_bgr, panel_x, panel_y + 4 * gap, bar_w, bar_h, 1.0 if mode_arp else 0.0, "ARP")

            # Control-hand bar (analog)
            ctrl01_for_bar = float(ctrl_pinky01) if 'ctrl_pinky01' in locals() and ctrl_pinky01 is not None else 0.0
            ctrl_label = "ctrl tempo" if mode_arp else "ctrl reverb"
            draw_bar(frame_bgr, panel_x, panel_y + 5 * gap, bar_w, bar_h, ctrl01_for_bar, ctrl_label)

            info_x = panel_x + bar_w + 95
            info_y0 = panel_y + 10
            mode_name = "ARP" if mode_arp else "PAD"
            rec_flag = "REC" if recorder.recording else "   "
            info_color = TOP_UI_COLOR

            cv2.putText(frame_bgr, f"{chord.name}  zone {int(active_zone)+1}/{NUM_ZONES}  tr {transpose:+d}  [{mode_name}]  {rec_flag}",
                        (info_x, info_y0), UI_FONT, UI_SCALE, info_color, UI_THICKNESS, cv2.LINE_AA)
            cv2.putText(frame_bgr, f"fist_raw {last_fist_raw:.3f}  fist_logic {float(ema_fist_logic):.3f}  expr_cap {expr_cap}  vel {vel}  preset {preset.name}",
                        (info_x, info_y0 + 18), UI_FONT, UI_SCALE, info_color, UI_THICKNESS, cv2.LINE_AA)
            cv2.putText(frame_bgr, (f"reverb {last_cc91}   chorus {last_cc93}" + (f"   tempo {arp_step_ms_curr}ms" if mode_arp else "")),
                        (info_x, info_y0 + 36), UI_FONT, UI_SCALE, info_color, UI_THICKNESS, cv2.LINE_AA)

            # Fist display near bbox (RAW only)
            if SHOW_HAND_BOX and main_lms is not None:
                xs = [lm.x for lm in main_lms]
                ys = [lm.y for lm in main_lms]
                min_x, max_x = int(min(xs) * w), int(max(xs) * w)
                min_y, max_y = int(min(ys) * h), int(max(ys) * h)
                anchor_x = max_x + 6
                anchor_y = max_y + 18
                anchor_x = int(np.clip(anchor_x, 6, w - 160))
                anchor_y = int(np.clip(anchor_y, 18, h - 6))
                draw_text_plain(frame_bgr, f"fist_raw {last_fist_raw:.3f}", (anchor_x, anchor_y), color=ORANGE, scale=0.50, thickness=1)

            # Small tint for zone (blue-ish)
            # tint zone strip lightly
            strip_y0 = int(h * 0.92)
            strip_y1 = int(h * 0.985)
            z_w = int((w - 2 * int(w * ZONE_MARGIN_X)) / NUM_ZONES)
            x0 = int(w * ZONE_MARGIN_X) + active_zone * z_w
            x1 = x0 + z_w
            overlay = frame_bgr.copy()
            cv2.rectangle(overlay, (x0, strip_y0), (x1, strip_y1), (255, 80, 40), -1)  # slight blue tint
            cv2.addWeighted(overlay, 0.10, frame_bgr, 0.90, 0, frame_bgr)

            # Debug payload
            last_result_data = {
                "t_ms": now_ms,
                "mode": "ARP" if mode_arp else "PAD",
                "zone": int(active_zone),
                "transpose": int(transpose),
                "spread": float(ema_spread),
                "spread01": float(spread01),
                "palm_scale": float(ema_palm),
                "fist_raw": float(last_fist_raw),
                "fist_logic": float(ema_fist_logic),
                "expr": int(last_expr_sent),
                "mod": int(mod_val),
                "reverb": int(last_cc91),
            }

            recorder.log(last_result_data)

        # handle keys (optional)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"):
            recorder.toggle()
        if key == ord("p"):
            preset_idx = (preset_idx + 1) % len(SOUND_PRESETS)
            preset = SOUND_PRESETS[preset_idx]
            # update base FX
            for ch in (CH_BASE, CH_MELODY):
                cc(ch, CC_REVERB, preset.base_reverb)
                cc(ch, CC_CHORUS, preset.base_chorus)
            last_cc91 = preset.base_reverb
            last_cc93 = preset.base_chorus

        cv2.imshow(WIN_NAME, frame_bgr)
        if cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

        frame_id += 1
        prev_time = time.time()

    # cleanup
    stop_all_now()
    grabber.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
