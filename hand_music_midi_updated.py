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

# ----------------------------
# Config
# ----------------------------
CAMERA_INDEX = 0
USE_DSHOW_ON_WINDOWS = True

MODEL_FILENAME = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

MAX_NUM_HANDS = 1

# Hold chord briefly if tracking drops
HAND_LOST_GRACE_MS = 900

# Zones / smoothing
# NOTE: Lower = less latency. Big values make note changes feel delayed.
ZONE_HOLD_MS = 90
EMA_ALPHA_Y = 0.35
EMA_ALPHA_PALM = 0.30
EMA_ALPHA_SPREAD = 0.35
EMA_ALPHA_FIST = 0.25

# Default vertical calibration (overridden by keys 3/4)
DEFAULT_Y_BOTTOM = 0.92
DEFAULT_Y_TOP = 0.20

# Volume (palm scale)
CHANNEL_VOLUME_CC7 = 127
MIN_EXPR = 30
MAX_EXPR = 127
NOTE_VELOCITY = 108

# Vibrato
VIB_RATE_HZ = 5.0
VIB_MAX_DEPTH = 2400        # pitchwheel units (max is 8191)
PITCHWHEEL_DEADBAND = 40

# Spread: thumb tip (4) -> pinky tip (20), normalized by palm scale
# These are thresholds on the *0..1 spread bar* (after calibration).
SPREAD_BAR_START = 0.12     # below this -> no vibrato
SPREAD_BAR_FULL = 0.78      # at/above -> max vibrato

# Slur / overlap (prevents hard restarts on chord changes)
SLUR_OVERLAP_MS = 140

# Fist / release fade-out
FIST_ON = 0.72
FIST_OFF = 0.60            # hysteresis
RELEASE_MS = 1000          # fade duration (ms)

# MIDI
MIDI_CHANNEL = 0
MIDI_PORT_SUBSTRING = None
SEND_REVERB_AND_CHORUS = True
SEND_PROGRAM_CHANGE = True
MIDI_PROGRAM = 89  # GM Pad 2 (warm) (0..127)

# Visual overlays
DRAW_SKELETON = True
SHOW_HAND_BOX = False
SHOW_ZONE_GUIDES = True
SHOW_PALM_POINT = False

# Window
WIN_NAME = "Hand Music (MIDI)"
ALLOW_WINDOW_RESIZE = True

# Camera / performance tuning
# Lowering resolution often boosts FPS.
REQUEST_CAM_W = 640
REQUEST_CAM_H = 480
# Run MediaPipe inference on a smaller image (display still uses full camera frame).
INFER_MAX_W = 640

HAND_CONNECTIONS: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17), (5, 13), (9, 17), (0, 9)
]

# ----------------------------
# Chords
# ----------------------------
@dataclass
class Chord:
    name: str
    notes: List[int]

CHORDS: List[Chord] = [
    Chord("D(add9/6)",      [50, 57, 62, 64, 66]),
    Chord("Em11(no3)",      [52, 59, 62, 69, 71]),
    Chord("G(add9/6)",      [55, 62, 67, 69, 71]),
    Chord("Asus2(add4)",    [57, 64, 69, 71, 74]),
    Chord("Bm11",           [59, 66, 69, 74, 76]),
]

# ----------------------------
# Helpers
# ----------------------------
def ensure_model_file(model_path: str) -> None:
    if os.path.exists(model_path):
        return
    print(f"[model] '{model_path}' not found. Downloading...")
    urllib.request.urlretrieve(MODEL_URL, model_path)
    print(f"[model] Downloaded to: {model_path}")

def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def clamp_int(v: float, lo: int, hi: int) -> int:
    v_int = int(round(v))
    return max(lo, min(hi, v_int))

def norm_to_pixel(lm, width: int, height: int) -> Tuple[int, int]:
    x_px = int(np.clip(lm.x, 0.0, 1.0) * width)
    y_px = int(np.clip(lm.y, 0.0, 1.0) * height)
    return x_px, y_px

def dist_norm(hand_landmarks, a: int, b: int) -> float:
    ax, ay = float(hand_landmarks[a].x), float(hand_landmarks[a].y)
    bx, by = float(hand_landmarks[b].x), float(hand_landmarks[b].y)
    return float(np.hypot(ax - bx, ay - by))

def palm_center_norm(hand_landmarks) -> Tuple[float, float]:
    idxs = [0, 5, 9, 13, 17]
    xs = [hand_landmarks[i].x for i in idxs]
    ys = [hand_landmarks[i].y for i in idxs]
    return float(np.mean(xs)), float(np.mean(ys))

def hand_bbox_norm(hand_landmarks) -> Tuple[float, float, float, float]:
    xs = [lm.x for lm in hand_landmarks]
    ys = [lm.y for lm in hand_landmarks]
    return float(np.min(xs)), float(np.min(ys)), float(np.max(xs)), float(np.max(ys))

def palm_scale_norm(hand_landmarks) -> float:
    # Wrist (0) -> Middle MCP (9): distance proxy
    return dist_norm(hand_landmarks, 0, 9)

def spread_thumb_pinky_norm(hand_landmarks, palm_scale: float) -> float:
    # Thumb tip (4) -> pinky tip (20), normalized by palm scale
    return dist_norm(hand_landmarks, 4, 20) / (palm_scale + 1e-6)

def fistness_norm(hand_landmarks, palm_scale: float) -> float:
    # 0..1: higher means more "closed"
    cx, cy = palm_center_norm(hand_landmarks)
    tips = [4, 8, 12, 16, 20]
    ds = []
    for i in tips:
        dx = float(hand_landmarks[i].x) - cx
        dy = float(hand_landmarks[i].y) - cy
        ds.append(float(np.hypot(dx, dy)))
    avg_tip = float(np.mean(ds))
    openish = (avg_tip / (palm_scale + 1e-6))
    return clamp01(1.0 - (openish / 1.8))

def draw_hand_skeleton(img_bgr: np.ndarray, hand_landmarks, w: int, h: int) -> None:
    pts = [norm_to_pixel(lm, w, h) for lm in hand_landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(img_bgr, pts[a], pts[b], (0, 255, 0), 2)
    for idx, (x, y) in enumerate(pts):
        cv2.circle(img_bgr, (x, y), 4, (0, 0, 255), -1)
        cv2.putText(img_bgr, str(idx), (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

def draw_hand_box(frame_bgr: np.ndarray, w: int, h: int, hand_landmarks) -> None:
    min_x, min_y, max_x, max_y = hand_bbox_norm(hand_landmarks)
    x1 = int(np.clip(min_x, 0.0, 1.0) * w)
    y1 = int(np.clip(min_y, 0.0, 1.0) * h)
    x2 = int(np.clip(max_x, 0.0, 1.0) * w)
    y2 = int(np.clip(max_y, 0.0, 1.0) * h)
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 200, 255), 2)

def draw_guides(frame_bgr: np.ndarray, w: int, h: int, y_top: float, y_bottom: float, num_zones: int) -> None:
    if y_bottom < y_top:
        y_top, y_bottom = y_bottom, y_top
    y_top_px = int(np.clip(y_top, 0.0, 1.0) * h)
    y_bottom_px = int(np.clip(y_bottom, 0.0, 1.0) * h)
    cv2.line(frame_bgr, (0, y_top_px), (w, y_top_px), (255, 255, 255), 2)
    cv2.line(frame_bgr, (0, y_bottom_px), (w, y_bottom_px), (255, 255, 255), 2)
    span = max(1, y_bottom_px - y_top_px)
    for i in range(1, num_zones):
        y = y_bottom_px - int(round(i * span / num_zones))
        cv2.line(frame_bgr, (0, y), (w, y), (255, 255, 0), 1)

# ----------------------------
# MIDI
# ----------------------------
def open_midi_out() -> mido.ports.BaseOutput:
    names = mido.get_output_names()
    if not names:
        raise RuntimeError("No MIDI output ports found. Install a MIDI synth/driver, then retry.")

    print("\nMIDI outputs found:")
    for n in names:
        print(" -", n)

    chosen = None
    if MIDI_PORT_SUBSTRING:
        for n in names:
            if MIDI_PORT_SUBSTRING.lower() in n.lower():
                chosen = n
                break
        if not chosen:
            print(f"\n[warn] No port matched substring '{MIDI_PORT_SUBSTRING}'. Using first port.")
            chosen = names[0]
    else:
        chosen = names[0]

    print("\nUsing MIDI out:", chosen)
    return mido.open_output(chosen)

def midi_cc(outport: mido.ports.BaseOutput, control: int, value: int) -> None:
    outport.send(mido.Message("control_change", channel=MIDI_CHANNEL, control=int(control), value=int(value)))

def midi_all_notes_off(outport: mido.ports.BaseOutput) -> None:
    midi_cc(outport, 123, 0)

def midi_set_expr(outport: mido.ports.BaseOutput, expr: int) -> None:
    midi_cc(outport, 11, int(expr))

def midi_set_channel_volume(outport: mido.ports.BaseOutput, vol: int) -> None:
    midi_cc(outport, 7, int(vol))

def midi_set_reverb_chorus(outport: mido.ports.BaseOutput) -> None:
    midi_cc(outport, 91, 100)  # reverb
    midi_cc(outport, 93, 78)   # chorus

def midi_program_change(outport: mido.ports.BaseOutput, program: int) -> None:
    outport.send(mido.Message("program_change", channel=MIDI_CHANNEL, program=int(program)))

def midi_pitchwheel(outport: mido.ports.BaseOutput, value: int) -> None:
    outport.send(mido.Message("pitchwheel", channel=MIDI_CHANNEL, pitch=int(value)))

def note_on(outport: mido.ports.BaseOutput, note: int, velocity: int) -> None:
    outport.send(mido.Message("note_on", channel=MIDI_CHANNEL, note=int(note), velocity=int(velocity)))

def note_off(outport: mido.ports.BaseOutput, note: int) -> None:
    outport.send(mido.Message("note_off", channel=MIDI_CHANNEL, note=int(note), velocity=0))

# ----------------------------
# UI Drawing
# ----------------------------
def draw_bar(
    img: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    value01: float,
    label: str,
    label_scale: float = 0.45,
) -> None:
    value01 = clamp01(value01)
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 1)
    fill = int(round(value01 * (w - 2)))
    if fill > 0:
        cv2.rectangle(img, (x + 1, y + 1), (x + 1 + fill, y + h - 1), (255, 255, 255), -1)
    cv2.putText(img, label, (x + w + 8, y + h - 2),
                cv2.FONT_HERSHEY_SIMPLEX, label_scale, (255, 255, 255), 1, cv2.LINE_AA)

def draw_rotated_text(img: np.ndarray, text: str, center: Tuple[int, int], angle_deg: float, scale: float = 0.45) -> None:
    # Text onto patch -> rotate -> overlay
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad = 8
    patch_w = tw + pad * 2
    patch_h = th + baseline + pad * 2
    patch = np.zeros((patch_h, patch_w, 3), dtype=np.uint8)
    cv2.putText(patch, text, (pad, pad + th), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    M = cv2.getRotationMatrix2D((patch_w / 2, patch_h / 2), angle_deg, 1.0)
    rotated = cv2.warpAffine(patch, M, (patch_w, patch_h),
                             flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))

    cx, cy = center
    x1 = int(cx - patch_w / 2)
    y1 = int(cy - patch_h / 2)
    x2 = x1 + patch_w
    y2 = y1 + patch_h

    H, W = img.shape[:2]
    sx1, sy1 = max(0, x1), max(0, y1)
    sx2, sy2 = min(W, x2), min(H, y2)
    if sx1 >= sx2 or sy1 >= sy2:
        return

    rx1, ry1 = sx1 - x1, sy1 - y1
    rx2, ry2 = rx1 + (sx2 - sx1), ry1 + (sy2 - sy1)

    roi = img[sy1:sy2, sx1:sx2]
    patch_roi = rotated[ry1:ry2, rx1:rx2]

    mask = (patch_roi.sum(axis=2) > 0)[:, :, None]
    roi[:] = np.where(mask, np.maximum(roi, patch_roi), roi)

# ----------------------------
# Capture thread (reduces latency)
# ----------------------------
class FrameGrabber:
    def __init__(self, cap: cv2.VideoCapture):
        self.cap = cap
        self.lock = threading.Lock()
        self.latest: Optional[np.ndarray] = None
        self.latest_t: float = 0.0
        self.stopped = False
        self.th = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.th.start()

    def _run(self) -> None:
        while not self.stopped:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self.lock:
                self.latest = frame
                self.latest_t = time.time()

    def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        with self.lock:
            if self.latest is None:
                return False, None, 0.0
            return True, self.latest.copy(), self.latest_t

    def stop(self) -> None:
        self.stopped = True

def maybe_downscale_for_inference(frame_bgr: np.ndarray) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    if INFER_MAX_W is None or w <= INFER_MAX_W:
        return frame_bgr
    scale = INFER_MAX_W / float(w)
    nh = int(round(h * scale))
    return cv2.resize(frame_bgr, (INFER_MAX_W, nh), interpolation=cv2.INTER_AREA)

# ----------------------------
# Main
# ----------------------------
def main() -> None:
    model_path = os.path.join(os.path.dirname(__file__), MODEL_FILENAME)
    ensure_model_file(model_path)

    outport = open_midi_out()
    midi_all_notes_off(outport)

    if SEND_PROGRAM_CHANGE:
        midi_program_change(outport, MIDI_PROGRAM)

    midi_set_channel_volume(outport, CHANNEL_VOLUME_CC7)
    midi_set_expr(outport, 110)
    if SEND_REVERB_AND_CHORUS:
        midi_set_reverb_chorus(outport)

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    HandLandmarker = mp.tasks.vision.HandLandmarker

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=MAX_NUM_HANDS,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap_flags = cv2.CAP_DSHOW if USE_DSHOW_ON_WINDOWS else 0
    cap = cv2.VideoCapture(CAMERA_INDEX, cap_flags)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try a different CAMERA_INDEX.")

    # Try to reduce internal buffering / raise FPS.
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    if REQUEST_CAM_W:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, REQUEST_CAM_W)
    if REQUEST_CAM_H:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUEST_CAM_H)

    grabber = FrameGrabber(cap)
    grabber.start()

    if ALLOW_WINDOW_RESIZE:
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    else:
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_AUTOSIZE)

    print("\nControls:")
    print("  q = quit")
    print("  p = print debug JSON")
    print("  1 = set FAR calibration (quiet) from current palm scale")
    print("  2 = set NEAR calibration (loud) from current palm scale")
    print("  3 = set LOW hand position (bottom of your zone range)")
    print("  4 = set HIGH hand position (top of your zone range)")
    print("  r = reset volume + height + spread calibration")
    print("  5 = set SPREAD MIN from current thumb-pinky (for bar + vibrato)")
    print("  6 = set SPREAD MAX from current thumb-pinky (for bar + vibrato)\n")

    ema_y: Optional[float] = None
    ema_palm: Optional[float] = None
    ema_spread: Optional[float] = None
    ema_fist: Optional[float] = None

    palm_min: Optional[float] = None
    palm_max: Optional[float] = None
    last_palm_seen: Optional[float] = None

    spread_min: Optional[float] = None
    spread_max: Optional[float] = None
    last_spread_seen: Optional[float] = None

    y_bottom: float = DEFAULT_Y_BOTTOM
    y_top: float = DEFAULT_Y_TOP

    num_zones = len(CHORDS)

    active_zone: Optional[int] = None
    candidate_zone: Optional[int] = None
    candidate_since_ms: Optional[int] = None

    held_notes: Set[int] = set()
    desired_notes: Set[int] = set()
    pending_off: List[Tuple[int, int]] = []  # (due_ms, note)

    last_expr_sent: Optional[int] = None
    last_pitchwheel_sent: int = 0

    last_result_data: Dict[str, Any] = {}
    last_hand_seen_ms: Optional[int] = None

    # Release state
    releasing: bool = False
    release_start_ms: int = 0
    release_start_expr: int = 0
    fist_is_on: bool = False

    # FPS
    fps_ema: Optional[float] = None
    last_loop_t = time.time()

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame_bgr, _ = grabber.read()
            if not ok or frame_bgr is None:
                time.sleep(0.005)
                continue

            now_sec = time.time()
            now_ms = int(now_sec * 1000)

            # FPS estimate
            dt = max(1e-6, now_sec - last_loop_t)
            last_loop_t = now_sec
            inst_fps = 1.0 / dt
            fps_ema = inst_fps if fps_ema is None else (0.10 * inst_fps + 0.90 * fps_ema)

            h, w = frame_bgr.shape[:2]

            # Process scheduled note-offs (slur overlap)
            if pending_off:
                still_pending = []
                for due_ms, n in pending_off:
                    if now_ms >= due_ms:
                        if n not in desired_notes and n in held_notes:
                            note_off(outport, n)
                            held_notes.discard(n)
                    else:
                        still_pending.append((due_ms, n))
                pending_off = still_pending

            # Inference
            infer_bgr = maybe_downscale_for_inference(frame_bgr)
            infer_rgb = cv2.cvtColor(infer_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=infer_rgb)
            result = landmarker.detect_for_video(mp_image, now_ms)

            hand_present = bool(result and result.hand_landmarks and len(result.hand_landmarks) > 0)
            grace_ok = (last_hand_seen_ms is not None) and ((now_ms - last_hand_seen_ms) <= HAND_LOST_GRACE_MS)

            if SHOW_ZONE_GUIDES:
                draw_guides(frame_bgr, w, h, y_top, y_bottom, num_zones)

            if hand_present:
                last_hand_seen_ms = now_ms
                hand_lms = result.hand_landmarks[0]

                if DRAW_SKELETON:
                    draw_hand_skeleton(frame_bgr, hand_lms, w, h)
                if SHOW_HAND_BOX:
                    draw_hand_box(frame_bgr, w, h, hand_lms)

                cx, cy = palm_center_norm(hand_lms)
                ps = palm_scale_norm(hand_lms)
                last_palm_seen = ps

                spread_raw = dist_norm(hand_lms, 4, 20)
                spread_norm = spread_thumb_pinky_norm(hand_lms, ps)
                last_spread_seen = spread_norm

                fistness = fistness_norm(hand_lms, ps)

                # EMA smoothing
                ema_y = cy if ema_y is None else (EMA_ALPHA_Y * cy + (1.0 - EMA_ALPHA_Y) * ema_y)
                ema_palm = ps if ema_palm is None else (EMA_ALPHA_PALM * ps + (1.0 - EMA_ALPHA_PALM) * ema_palm)
                ema_spread = spread_norm if ema_spread is None else (EMA_ALPHA_SPREAD * spread_norm + (1.0 - EMA_ALPHA_SPREAD) * ema_spread)
                ema_fist = fistness if ema_fist is None else (EMA_ALPHA_FIST * fistness + (1.0 - EMA_ALPHA_FIST) * ema_fist)

                # Spread calibration bootstrap + gentle auto-widening
                if spread_min is None or spread_max is None:
                    spread_min = float(ema_spread) * 0.80
                    spread_max = float(ema_spread) * 1.25
                else:
                    spread_min = min(spread_min * 0.9995 + 0.0005 * float(ema_spread), float(ema_spread))
                    spread_max = max(spread_max * 0.9995 + 0.0005 * float(ema_spread), float(ema_spread))

                # Default palm calibration bootstrap
                if palm_min is None or palm_max is None:
                    palm_min = float(ema_palm) * 0.85
                    palm_max = float(ema_palm) * 1.25

                if y_bottom < y_top:
                    y_top, y_bottom = y_bottom, y_top

                # Fist latch with hysteresis
                if not fist_is_on and float(ema_fist) >= FIST_ON:
                    fist_is_on = True
                elif fist_is_on and float(ema_fist) <= FIST_OFF:
                    fist_is_on = False

                # Start release fade on fist close
                if fist_is_on and not releasing:
                    releasing = True
                    release_start_ms = now_ms
                    release_start_expr = int(last_expr_sent if last_expr_sent is not None else 110)
                    if last_pitchwheel_sent != 0:
                        midi_pitchwheel(outport, 0)
                        last_pitchwheel_sent = 0

                # Cancel release if fist opens early
                if (not fist_is_on) and releasing:
                    releasing = False

                # Zone mapping (freeze switching during release)
                denom = (y_bottom - y_top)
                denom = denom if abs(denom) > 1e-6 else 1e-6
                t_top = clamp01((y_bottom - float(ema_y)) / denom)
                raw_zone = int(np.clip(t_top * num_zones, 0, num_zones - 1))

                if active_zone is None:
                    active_zone = raw_zone
                    candidate_zone = None
                    candidate_since_ms = None
                elif not releasing:
                    if raw_zone != active_zone:
                        if candidate_zone != raw_zone:
                            candidate_zone = raw_zone
                            candidate_since_ms = now_ms
                        else:
                            if candidate_since_ms is not None and (now_ms - candidate_since_ms) >= ZONE_HOLD_MS:
                                active_zone = candidate_zone
                                candidate_zone = None
                                candidate_since_ms = None
                    else:
                        candidate_zone = None
                        candidate_since_ms = None

                chord = CHORDS[int(active_zone)] if active_zone is not None else CHORDS[0]
                if not releasing:
                    desired_notes = set(chord.notes)

                # Expression: normal mapping unless releasing
                if releasing:
                    t = clamp01((now_ms - release_start_ms) / float(RELEASE_MS))
                    expr = int(round((1.0 - t) * release_start_expr))
                else:
                    lo = float(min(palm_min, palm_max))
                    hi = float(max(palm_min, palm_max))
                    if abs(hi - lo) < 1e-6:
                        expr = 118
                    else:
                        x = float(np.clip((float(ema_palm) - lo) / (hi - lo), 0.0, 1.0))
                        expr = clamp_int(MIN_EXPR + x * (MAX_EXPR - MIN_EXPR), 0, 127)

                if last_expr_sent is None or abs(expr - last_expr_sent) >= 2:
                    midi_set_expr(outport, expr)
                    last_expr_sent = expr

                # Notes: slur overlap on chord changes
                to_on = sorted(list(desired_notes - held_notes))
                to_off = sorted(list(held_notes - desired_notes))
                for n in to_on:
                    note_on(outport, n, NOTE_VELOCITY)
                    held_notes.add(n)
                for n in to_off:
                    pending_off.append((now_ms + SLUR_OVERLAP_MS, n))

                # Finish release: stop notes cleanly
                if releasing and (now_ms - release_start_ms) >= RELEASE_MS:
                    if last_expr_sent != 0:
                        midi_set_expr(outport, 0)
                        last_expr_sent = 0
                    for n in sorted(list(held_notes)):
                        note_off(outport, n)
                    held_notes.clear()
                    desired_notes.clear()
                    releasing = False

                # Spread bar normalization
                s_lo = float(min(spread_min, spread_max))
                s_hi = float(max(spread_min, spread_max))
                spread01 = 0.0 if abs(s_hi - s_lo) < 1e-6 else clamp01((float(ema_spread) - s_lo) / (s_hi - s_lo))

                # Vibrato depth (disabled during release)
                depth = 0
                if not releasing:
                    if spread01 <= SPREAD_BAR_START:
                        depth = 0
                    elif spread01 >= SPREAD_BAR_FULL:
                        depth = VIB_MAX_DEPTH
                    else:
                        frac = (spread01 - SPREAD_BAR_START) / (SPREAD_BAR_FULL - SPREAD_BAR_START)
                        depth = int(round(frac * VIB_MAX_DEPTH))

                    vib = int(round(np.sin(2.0 * np.pi * VIB_RATE_HZ * now_sec) * depth))
                    if abs(vib - last_pitchwheel_sent) >= PITCHWHEEL_DEADBAND:
                        midi_pitchwheel(outport, vib)
                        last_pitchwheel_sent = vib
                else:
                    if last_pitchwheel_sent != 0:
                        midi_pitchwheel(outport, 0)
                        last_pitchwheel_sent = 0

                # Thumb<->pinky line + rotated text along the line
                p4 = norm_to_pixel(hand_lms[4], w, h)
                p20 = norm_to_pixel(hand_lms[20], w, h)
                cv2.line(frame_bgr, p4, p20, (255, 255, 255), 2)

                mx = int(round((p4[0] + p20[0]) / 2))
                my = int(round((p4[1] + p20[1]) / 2))
                dx = float(p20[0] - p4[0])
                dy = float(p20[1] - p4[1])
                angle = float(np.degrees(np.arctan2(dy, dx)))
                draw_rotated_text(
                    frame_bgr,
                    f"spread 4-20: {ema_spread:.3f}  ({int(round(spread01*100))}%)",
                    (mx, my),
                    angle,
                    scale=0.45
                )

                # Mini bars + compact text UI
                panel_x, panel_y = 10, 10
                bar_w, bar_h, gap = 140, 10, 14

                draw_bar(frame_bgr, panel_x, panel_y + 0 * gap, bar_w, bar_h, spread01, "spread", 0.45)
                draw_bar(frame_bgr, panel_x, panel_y + 1 * gap, bar_w, bar_h, float(ema_fist), "fist", 0.45)
                expr01 = clamp01((float(last_expr_sent) - MIN_EXPR) / float(max(1, MAX_EXPR - MIN_EXPR)))
                draw_bar(frame_bgr, panel_x, panel_y + 2 * gap, bar_w, bar_h, expr01, "expr", 0.45)
                vib01 = clamp01(depth / float(VIB_MAX_DEPTH)) if VIB_MAX_DEPTH > 0 else 0.0
                draw_bar(frame_bgr, panel_x, panel_y + 3 * gap, bar_w, bar_h, vib01, "vib", 0.45)

                text_y = panel_y + 4 * gap + 18
                cv2.putText(frame_bgr,
                            f"{chord.name}  zone {int(active_zone)+1}/{num_zones}" + ("  (RELEASE)" if releasing else ""),
                            (10, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame_bgr,
                            f"fps {fps_ema:.1f}  expr {last_expr_sent}  palm {ema_palm:.4f}  fist {ema_fist:.2f}",
                            (10, text_y + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

                last_result_data = {
                    "zone": int(active_zone),
                    "chord": chord.name,
                    "expr": int(last_expr_sent if last_expr_sent is not None else -1),
                    "palm_scale": float(ps),
                    "ema_palm": float(ema_palm),
                    "spread_raw_4_20": float(spread_raw),
                    "spread_norm_4_20": float(spread_norm),
                    "ema_spread": float(ema_spread),
                    "spread01": float(spread01),
                    "fistness": float(fistness),
                    "ema_fist": float(ema_fist),
                    "vib_depth": int(depth),
                    "pitchwheel": int(last_pitchwheel_sent),
                    "releasing": bool(releasing),
                }

            else:
                # No hand: hold briefly, else stop
                if not grace_ok and not releasing:
                    desired_notes = set()
                    if held_notes:
                        for n in sorted(list(held_notes)):
                            note_off(outport, n)
                        held_notes.clear()
                    if last_expr_sent is None or last_expr_sent != 0:
                        midi_set_expr(outport, 0)
                        last_expr_sent = 0
                    if last_pitchwheel_sent != 0:
                        midi_pitchwheel(outport, 0)
                        last_pitchwheel_sent = 0
                    last_result_data = {"hand_present": False, "grace_ok": False}
                    cv2.putText(frame_bgr, "No hand (stopped)", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    if last_pitchwheel_sent != 0:
                        midi_pitchwheel(outport, 0)
                        last_pitchwheel_sent = 0
                    last_result_data = {"hand_present": False, "grace_ok": True}
                    cv2.putText(frame_bgr, "No hand (holding)", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 255), 2, cv2.LINE_AA)

            # Display (resizable)
            display_frame = frame_bgr
            if ALLOW_WINDOW_RESIZE:
                try:
                    _, _, win_w, win_h = cv2.getWindowImageRect(WIN_NAME)
                    if win_w > 0 and win_h > 0 and (win_w != w or win_h != h):
                        display_frame = cv2.resize(frame_bgr, (win_w, win_h), interpolation=cv2.INTER_AREA)
                except Exception:
                    pass

            cv2.imshow(WIN_NAME, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("p"):
                print(json.dumps(last_result_data, indent=2))
            if key == ord("1") and last_palm_seen is not None:
                palm_min = float(last_palm_seen)
                print(f"[cal] Set FAR (min) palm_scale = {palm_min:.6f}")
            if key == ord("2") and last_palm_seen is not None:
                palm_max = float(last_palm_seen)
                print(f"[cal] Set NEAR (max) palm_scale = {palm_max:.6f}")
            if key == ord("3") and ema_y is not None:
                y_bottom = float(ema_y)
                print(f"[cal] Set LOW position (y_bottom) = {y_bottom:.3f}")
            if key == ord("4") and ema_y is not None:
                y_top = float(ema_y)
                print(f"[cal] Set HIGH position (y_top) = {y_top:.3f}")
            if key == ord("5") and last_spread_seen is not None:
                spread_min = float(last_spread_seen)
                print(f"[cal] Set SPREAD MIN = {spread_min:.6f}")
            if key == ord("6") and last_spread_seen is not None:
                spread_max = float(last_spread_seen)
                print(f"[cal] Set SPREAD MAX = {spread_max:.6f}")
            if key == ord("r"):
                palm_min = None
                palm_max = None
                y_bottom = DEFAULT_Y_BOTTOM
                y_top = DEFAULT_Y_TOP
                spread_min = None
                spread_max = None
                print("[cal] Reset volume + height + spread calibration")

    # Cleanup
    try:
        midi_all_notes_off(outport)
        midi_pitchwheel(outport, 0)
        midi_set_expr(outport, 0)
    except Exception:
        pass
    grabber.stop()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
