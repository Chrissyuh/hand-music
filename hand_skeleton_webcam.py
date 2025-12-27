import os
import time
import json
import urllib.request
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

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
NUM_ZONES = 5
ZONE_HOLD_FRAMES = 10
EMA_ALPHA_Y = 0.25
EMA_ALPHA_PALM = 0.25
EMA_ALPHA_SPREAD = 0.25

# Default vertical calibration (overridden by keys 3/4)
DEFAULT_Y_BOTTOM = 0.92
DEFAULT_Y_TOP = 0.20

# Volume (distance proxy)
CHANNEL_VOLUME_CC7 = 127
MIN_EXPR = 30
MAX_EXPR = 127
NOTE_VELOCITY = 108

# Vibrato from horizontal finger spacing (PIP x-range), normalized by palm scale (0-9)
VIB_RATE_HZ = 5.0

# Key change: make vibrato subtler + scalable
VIB_MAX_DEPTH = 500          # was ~2400; lower = less “full vibrato”
SPREAD_START = 0.60          # below this -> no vibrato
SPREAD_FULL = 0.98           # at/above this -> max vibrato
SPREAD_CURVE_POWER = 2.2     # >1 = gentler onset, more control at low spread

PITCHWHEEL_DEADBAND = 10     # smaller than before so it doesn't “stick”

# Force synth pitch bend range smaller (many synths honor this)
SET_PITCH_BEND_RANGE = True
PITCH_BEND_RANGE_SEMITONES = 1   # 1 = subtle; try 2 if you want more
PITCH_BEND_RANGE_CENTS = 0

# MIDI
MIDI_CHANNEL = 0
MIDI_PORT_SUBSTRING = None
SEND_REVERB_AND_CHORUS = True
SEND_PROGRAM_CHANGE = True
MIDI_PROGRAM = 89  # GM Pad 2 (warm) (0..127)

# Visual overlays
DRAW_SKELETON = True
SHOW_DEBUG_TEXT = True
SHOW_ZONE_GUIDES = True
SHOW_HAND_BOX = True
SHOW_PALM_POINT = True

HAND_CONNECTIONS: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17), (5, 13), (9, 17), (0, 9)
]

# ----------------------------
# Ethereal chord set (compatible-ish, voice-led)
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

def draw_hand_skeleton(img_bgr: np.ndarray, hand_landmarks, w: int, h: int) -> None:
    pts = [norm_to_pixel(lm, w, h) for lm in hand_landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(img_bgr, pts[a], pts[b], (0, 255, 0), 2)
    for idx, (x, y) in enumerate(pts):
        cv2.circle(img_bgr, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(img_bgr, str(idx), (x + 6, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

def palm_center_norm(hand_landmarks) -> Tuple[float, float]:
    idxs = [0, 5, 9, 13, 17]
    xs = [hand_landmarks[i].x for i in idxs]
    ys = [hand_landmarks[i].y for i in idxs]
    return float(np.mean(xs)), float(np.mean(ys))

def hand_bbox_norm(hand_landmarks) -> Tuple[float, float, float, float]:
    xs = [lm.x for lm in hand_landmarks]
    ys = [lm.y for lm in hand_landmarks]
    return float(np.min(xs)), float(np.min(ys)), float(np.max(xs)), float(np.max(ys))

def dist_norm(hand_landmarks, a: int, b: int) -> float:
    ax, ay = float(hand_landmarks[a].x), float(hand_landmarks[a].y)
    bx, by = float(hand_landmarks[b].x), float(hand_landmarks[b].y)
    return float(np.hypot(ax - bx, ay - by))

def palm_scale_norm(hand_landmarks) -> float:
    # Wrist (0) -> Middle MCP (9): changes strongly with distance, weakly with finger splay
    return dist_norm(hand_landmarks, 0, 9)

def finger_horizontal_spread_norm(hand_landmarks, palm_scale: float) -> float:
    # PIP joints x-range (6,10,14,18) normalized by palm scale
    pip_idxs = [6, 10, 14, 18]
    xs = [float(hand_landmarks[i].x) for i in pip_idxs]
    x_range = float(max(xs) - min(xs))
    return x_range / (palm_scale + 1e-6)

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

def midi_set_pitch_bend_range(outport: mido.ports.BaseOutput, semitones: int, cents: int = 0) -> None:
    """
    RPN 0,0 = Pitch Bend Sensitivity.
    Many synths support this; some ignore it.
    """
    # Select RPN 0,0
    midi_cc(outport, 101, 0)  # RPN MSB
    midi_cc(outport, 100, 0)  # RPN LSB
    # Data Entry
    midi_cc(outport, 6, int(semitones))   # semitones
    midi_cc(outport, 38, int(cents))      # cents
    # Deselect RPN
    midi_cc(outport, 101, 127)
    midi_cc(outport, 100, 127)

def chord_note_on(outport: mido.ports.BaseOutput, notes: List[int], velocity: int) -> None:
    for n in notes:
        outport.send(mido.Message("note_on", channel=MIDI_CHANNEL, note=int(n), velocity=int(velocity)))

def chord_note_off(outport: mido.ports.BaseOutput, notes: List[int]) -> None:
    for n in notes:
        outport.send(mido.Message("note_off", channel=MIDI_CHANNEL, note=int(n), velocity=0))

def draw_guides(frame_bgr: np.ndarray, w: int, h: int, y_top: float, y_bottom: float) -> None:
    if y_bottom < y_top:
        y_top, y_bottom = y_bottom, y_top

    y_top_px = int(np.clip(y_top, 0.0, 1.0) * h)
    y_bottom_px = int(np.clip(y_bottom, 0.0, 1.0) * h)

    cv2.line(frame_bgr, (0, y_top_px), (w, y_top_px), (255, 255, 255), 2)
    cv2.line(frame_bgr, (0, y_bottom_px), (w, y_bottom_px), (255, 255, 255), 2)

    span = max(1, y_bottom_px - y_top_px)
    for i in range(1, NUM_ZONES):
        y = y_bottom_px - int(round(i * span / NUM_ZONES))
        cv2.line(frame_bgr, (0, y), (w, y), (255, 255, 0), 1)

def draw_hand_box(frame_bgr: np.ndarray, w: int, h: int, hand_landmarks) -> None:
    min_x, min_y, max_x, max_y = hand_bbox_norm(hand_landmarks)
    x1 = int(np.clip(min_x, 0.0, 1.0) * w)
    y1 = int(np.clip(min_y, 0.0, 1.0) * h)
    x2 = int(np.clip(max_x, 0.0, 1.0) * w)
    y2 = int(np.clip(max_y, 0.0, 1.0) * h)
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 200, 255), 2)

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

    if SET_PITCH_BEND_RANGE:
        midi_set_pitch_bend_range(outport, PITCH_BEND_RANGE_SEMITONES, PITCH_BEND_RANGE_CENTS)
        midi_pitchwheel(outport, 0)

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

    print("\nControls:")
    print("  q = quit")
    print("  1 = set FAR calibration (quiet) from current palm scale")
    print("  2 = set NEAR calibration (loud) from current palm scale")
    print("  3 = set LOW hand position (bottom of your zone range)")
    print("  4 = set HIGH hand position (top of your zone range)")
    print("  r = reset calibrations")
    print("  p = print debug JSON\n")

    ema_y: Optional[float] = None
    ema_palm: Optional[float] = None
    ema_spread: Optional[float] = None

    palm_min: Optional[float] = None
    palm_max: Optional[float] = None
    last_palm_seen: Optional[float] = None

    y_bottom: float = DEFAULT_Y_BOTTOM
    y_top: float = DEFAULT_Y_TOP

    active_zone: Optional[int] = None
    pending_zone: Optional[int] = None
    pending_count = 0

    active_notes: List[int] = []
    last_expr_sent: Optional[int] = None
    last_pitchwheel_sent: int = 0

    last_result_data: Dict[str, Any] = {}
    last_hand_seen_ms: Optional[int] = None

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            h, w = frame_bgr.shape[:2]
            now_sec = time.time()
            now_ms = int(now_sec * 1000)

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = landmarker.detect_for_video(mp_image, now_ms)

            hand_present = bool(result and result.hand_landmarks and len(result.hand_landmarks) > 0)

            grace_ok = (
                (last_hand_seen_ms is not None)
                and ((now_ms - last_hand_seen_ms) <= HAND_LOST_GRACE_MS)
            )

            if SHOW_ZONE_GUIDES:
                draw_guides(frame_bgr, w, h, y_top, y_bottom)

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

                spread = finger_horizontal_spread_norm(hand_lms, ps)

                ema_y = cy if ema_y is None else (EMA_ALPHA_Y * cy + (1.0 - EMA_ALPHA_Y) * ema_y)
                ema_palm = ps if ema_palm is None else (EMA_ALPHA_PALM * ps + (1.0 - EMA_ALPHA_PALM) * ema_palm)
                ema_spread = spread if ema_spread is None else (EMA_ALPHA_SPREAD * spread + (1.0 - EMA_ALPHA_SPREAD) * ema_spread)

                if SHOW_PALM_POINT and ema_y is not None:
                    px = int(np.clip(cx, 0.0, 1.0) * w)
                    py = int(np.clip(float(ema_y), 0.0, 1.0) * h)
                    cv2.circle(frame_bgr, (px, py), 8, (255, 0, 255), -1)

                if palm_min is None or palm_max is None:
                    palm_min = float(ema_palm) * 0.85
                    palm_max = float(ema_palm) * 1.25

                if y_bottom < y_top:
                    y_top, y_bottom = y_bottom, y_top

                denom = (y_bottom - y_top)
                denom = denom if abs(denom) > 1e-6 else 1e-6
                t_top = clamp01((y_bottom - float(ema_y)) / denom)
                raw_zone = int(np.clip(t_top * NUM_ZONES, 0, NUM_ZONES - 1))

                if active_zone is None:
                    active_zone = raw_zone
                    pending_zone = None
                    pending_count = 0
                else:
                    if raw_zone != active_zone:
                        if pending_zone != raw_zone:
                            pending_zone = raw_zone
                            pending_count = 1
                        else:
                            pending_count += 1
                            if pending_count >= ZONE_HOLD_FRAMES:
                                active_zone = pending_zone
                                pending_zone = None
                                pending_count = 0
                    else:
                        pending_zone = None
                        pending_count = 0

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

                chord = CHORDS[int(active_zone)]
                new_notes = chord.notes
                if active_notes != new_notes:
                    old_set = set(active_notes)
                    new_set = set(new_notes)

                    to_off = sorted(list(old_set - new_set))
                    to_on = sorted(list(new_set - old_set))

                    if to_off:
                        chord_note_off(outport, to_off)
                    if to_on:
                        chord_note_on(outport, to_on, velocity=NOTE_VELOCITY)

                    active_notes = new_notes.copy()

                # ---- Vibrato depth mapping (gentle + controllable) ----
                s = float(ema_spread) if ema_spread is not None else 0.0

                if s <= SPREAD_START:
                    depth = 0
                elif s >= SPREAD_FULL:
                    depth = VIB_MAX_DEPTH
                else:
                    frac = (s - SPREAD_START) / (SPREAD_FULL - SPREAD_START)
                    frac = float(frac ** SPREAD_CURVE_POWER)  # curve the response
                    depth = int(round(frac * VIB_MAX_DEPTH))

                if depth == 0:
                    # Ensure we truly silence vibrato immediately
                    if last_pitchwheel_sent != 0:
                        midi_pitchwheel(outport, 0)
                        last_pitchwheel_sent = 0
                else:
                    vib = int(round(np.sin(2.0 * np.pi * VIB_RATE_HZ * now_sec) * depth))
                    if abs(vib - last_pitchwheel_sent) >= PITCHWHEEL_DEADBAND:
                        midi_pitchwheel(outport, vib)
                        last_pitchwheel_sent = vib

                last_result_data = {
                    "t_top": float(t_top),
                    "zone": int(active_zone),
                    "chord": chord.name,
                    "palm_scale": float(ps),
                    "ema_palm": float(ema_palm),
                    "expr": int(last_expr_sent if last_expr_sent is not None else -1),
                    "spread_x_norm": float(spread),
                    "ema_spread": float(ema_spread),
                    "vib_depth": int(depth),
                    "pitchwheel": int(last_pitchwheel_sent),
                }

                if SHOW_DEBUG_TEXT:
                    cv2.putText(frame_bgr, f"Chord: {chord.name} (zone {active_zone+1}/{NUM_ZONES})",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                    cv2.putText(frame_bgr, f"Expr:{last_expr_sent}  SpreadX:{ema_spread:.3f}  Depth:{depth}  BendRange:{PITCH_BEND_RANGE_SEMITONES}st",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            else:
                if not grace_ok:
                    if active_notes:
                        chord_note_off(outport, active_notes)
                        active_notes = []
                    if last_expr_sent is None or last_expr_sent != 0:
                        midi_set_expr(outport, 0)
                        last_expr_sent = 0
                    if last_pitchwheel_sent != 0:
                        midi_pitchwheel(outport, 0)
                        last_pitchwheel_sent = 0
                    last_result_data = {"hand_present": False, "grace_ok": False}
                else:
                    if last_pitchwheel_sent != 0:
                        midi_pitchwheel(outport, 0)
                        last_pitchwheel_sent = 0
                    last_result_data = {"hand_present": False, "grace_ok": True}

                if SHOW_DEBUG_TEXT:
                    msg = "No hand (holding)" if grace_ok else "No hand (stopped)"
                    cv2.putText(frame_bgr, msg, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("Hand Music (MIDI)", frame_bgr)

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
            if key == ord("r"):
                palm_min = None
                palm_max = None
                y_bottom = DEFAULT_Y_BOTTOM
                y_top = DEFAULT_Y_TOP
                print("[cal] Reset volume + height calibration")

    try:
        midi_all_notes_off(outport)
        midi_pitchwheel(outport, 0)
    except Exception:
        pass
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
