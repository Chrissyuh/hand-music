import time
import math
import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np
import mediapipe as mp
import mido

# ============================================================
# Hand Music (V2)
# - Faster + lower latency (buffer controls + optional downscale)
# - Slur-ish chord transitions (overlap on chord changes)
# - Spread = distance(thumb_tip(4), pinky_tip(20)) / palm_width
# - Draw spread line (thumb<->pinky) + small spread bar
# - Small HUD with tiny bars + small text
# - Fist triggers 1s fade-out (CC11), then notes stop
# - Window is resizable (WINDOW_NORMAL) and frame scales to window
# ============================================================

WINDOW_NAME = "Hand Music"

# ---------- Camera / perf ----------
CAM_INDEX = 0
CAM_BACKEND = cv2.CAP_DSHOW  # best on Windows; ignored on non-Windows builds
CAM_W = 640
CAM_H = 360
CAM_FPS = 60
CAP_BUFFER = 1               # try to keep latency low

INFER_DOWNSCALE = 1.0        # 1.0 = full res; 0.75 / 0.5 can boost fps
DROP_OLD_FRAME_EACH_LOOP = True  # quick way to reduce buffering on some cams

# ---------- MIDI ----------
MIDI_CHANNEL = 0
NOTE_VELOCITY = 108
CHANNEL_VOLUME_CC7 = 127
DEFAULT_EXPR = 110           # CC11 expression start value (0-127)
MIDI_PORT_SUBSTRING = None   # e.g. "Microsoft GS Wavetable Synth" or leave None

# ---------- Mapping ----------
NUM_ZONES = 8
ZONE_HOLD_MS = 90            # how long a zone must be stable before switching
CHORD_SLUR_MS = 110          # overlap old/new chord this long (ms)

# Volume based on vertical hand position (palm center): top = loud, bottom = quiet
Y_TOP_DEFAULT = 0.20
Y_BOTTOM_DEFAULT = 0.92
MIN_EXPR = 20
MAX_EXPR = 127

# Vibrato (pitch bend) based on spread_norm (0..1) with thresholds
VIB_RATE_HZ = 5.0
VIB_MAX_DEPTH = 2400         # pitchwheel units (max 8191)
SPREAD_START_N = 0.20        # start vibrato at 20% of calibrated spread range
SPREAD_FULL_N = 0.85         # max vibrato at 85% of calibrated spread range
PITCHWHEEL_DEADBAND = 40

# Fist detection ratio thresholds (scale-invariant)
# ratio = mean(dist(fingertips -> palm_center)) / palm_width
FIST_RATIO_ON = 0.78
FIST_RATIO_OFF = 0.88
FIST_RELEASE_MS = 1000

# ---------- UI ----------
FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.45
TEXT_THICK = 1
BAR_H = 8
BAR_W = 140
BAR_PAD = 6

# ---------- Chords (per zone) ----------
# Simple diatonic-ish chord map in C major-ish (you can customize)
ZONE_CHORDS: Dict[int, List[int]] = {
    0: [60, 64, 67],  # C
    1: [62, 65, 69],  # Dm
    2: [64, 67, 71],  # Em
    3: [65, 69, 72],  # F
    4: [67, 71, 74],  # G
    5: [69, 72, 76],  # Am
    6: [71, 74, 77],  # Bdim-ish
    7: [72, 76, 79],  # C (up)
}

# ============================================================
# Helpers
# ============================================================

def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

def dist2(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.hypot(dx, dy)

def choose_midi_port(names: List[str]) -> Optional[str]:
    if not names:
        return None
    # 1) explicit substring
    if MIDI_PORT_SUBSTRING:
        for n in names:
            if MIDI_PORT_SUBSTRING.lower() in n.lower():
                return n
    # 2) windows heuristic
    for key in ["Microsoft GS Wavetable Synth", "Wavetable", "Microsoft"]:
        for n in names:
            if key.lower() in n.lower():
                return n
    # 3) fallback
    return names[0]

def open_midi_out(port_name: Optional[str]) -> Tuple[Optional[mido.ports.BaseOutput], Optional[str], List[str]]:
    names = mido.get_output_names()
    if not names:
        return None, None, []
    chosen = port_name or choose_midi_port(names)
    out = mido.open_output(chosen)
    return out, chosen, names

def midi_cc(outport: mido.ports.BaseOutput, control: int, value: int) -> None:
    outport.send(mido.Message("control_change", channel=MIDI_CHANNEL, control=int(control), value=int(value)))

def midi_set_expr(outport: mido.ports.BaseOutput, expr: int) -> None:
    expr = max(0, min(127, int(expr)))
    midi_cc(outport, 11, expr)

def midi_set_channel_volume(outport: mido.ports.BaseOutput, vol: int) -> None:
    vol = max(0, min(127, int(vol)))
    midi_cc(outport, 7, vol)

def midi_all_notes_off(outport: mido.ports.BaseOutput) -> None:
    midi_cc(outport, 123, 0)

def midi_pitchwheel(outport: mido.ports.BaseOutput, pitch: int) -> None:
    pitch = max(-8192, min(8191, int(pitch)))
    outport.send(mido.Message("pitchwheel", channel=MIDI_CHANNEL, pitch=pitch))

def chord_note_on(outport: mido.ports.BaseOutput, notes: List[int], velocity: int) -> None:
    for n in notes:
        outport.send(mido.Message("note_on", channel=MIDI_CHANNEL, note=int(n), velocity=int(velocity)))

def chord_note_off(outport: mido.ports.BaseOutput, notes: List[int]) -> None:
    for n in notes:
        outport.send(mido.Message("note_off", channel=MIDI_CHANNEL, note=int(n), velocity=0))

def draw_small_bar(frame: np.ndarray, x: int, y: int, label: str, value01: float,
                   color: Tuple[int,int,int], show_line_at: Optional[float]=None) -> int:
    """Draw a tiny bar with optional vertical line marker. Returns next y."""
    value01 = clamp01(value01)
    cv2.putText(frame, label, (x, y-2), FONT, TEXT_SCALE, (235,235,235), TEXT_THICK, cv2.LINE_AA)

    bx = x
    by = y + 4
    cv2.rectangle(frame, (bx, by), (bx+BAR_W, by+BAR_H), (40,40,40), -1)
    fill = int(BAR_W * value01)
    if fill > 0:
        cv2.rectangle(frame, (bx, by), (bx+fill, by+BAR_H), color, -1)

    if show_line_at is not None:
        lx = bx + int(BAR_W * clamp01(show_line_at))
        cv2.line(frame, (lx, by-2), (lx, by+BAR_H+2), (255,255,255), 1, cv2.LINE_AA)

    cv2.rectangle(frame, (bx, by), (bx+BAR_W, by+BAR_H), (90,90,90), 1)
    return by + BAR_H + BAR_PAD

def put_text_small(frame: np.ndarray, x: int, y: int, text: str, color=(235,235,235)) -> int:
    cv2.putText(frame, text, (x,y), FONT, TEXT_SCALE, color, TEXT_THICK, cv2.LINE_AA)
    return y + int(16*TEXT_SCALE) + 6

def draw_rotated_text(frame: np.ndarray, text: str, center: Tuple[int,int], angle_deg: float, color=(255,255,255)) -> None:
    """Draw rotated text by rendering onto a small canvas then warping."""
    (tw, th), base = cv2.getTextSize(text, FONT, TEXT_SCALE, TEXT_THICK)
    pad = 6
    w = tw + pad*2
    h = th + base + pad*2
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(canvas, text, (pad, h-pad-base), FONT, TEXT_SCALE, color, TEXT_THICK, cv2.LINE_AA)

    M = cv2.getRotationMatrix2D((w/2, h/2), angle_deg, 1.0)
    rot = cv2.warpAffine(canvas, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

    gray = cv2.cvtColor(rot, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    cx, cy = center
    x0 = cx - w//2
    y0 = cy - h//2

    H, W = frame.shape[:2]
    x1 = max(0, x0)
    y1 = max(0, y0)
    x2 = min(W, x0+w)
    y2 = min(H, y0+h)
    if x1 >= x2 or y1 >= y2:
        return

    rx1 = x1 - x0
    ry1 = y1 - y0
    rx2 = rx1 + (x2 - x1)
    ry2 = ry1 + (y2 - y1)

    roi = frame[y1:y2, x1:x2]
    rot_roi = rot[ry1:ry2, rx1:rx2]
    mask_roi = mask[ry1:ry2, rx1:rx2]

    inv = cv2.bitwise_not(mask_roi)
    bg = cv2.bitwise_and(roi, roi, mask=inv)
    fg = cv2.bitwise_and(rot_roi, rot_roi, mask=mask_roi)
    frame[y1:y2, x1:x2] = cv2.add(bg, fg)

# ============================================================
# Capture thread to reduce camera buffering latency
# ============================================================

class FrameGrabber:
    def __init__(self, cap: cv2.VideoCapture):
        self.cap = cap
        self.lock = threading.Lock()
        self.latest: Optional[np.ndarray] = None
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self.lock:
                self.latest = frame

    def get(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.latest is None:
                return None
            return self.latest.copy()

    def stop(self):
        self.running = False
        self.thread.join(timeout=1.0)

# ============================================================
# Main
# ============================================================

@dataclass
class ZoneDwell:
    active_zone: Optional[int] = None
    pending_zone: Optional[int] = None
    pending_since_ms: int = 0

    def update(self, raw_zone: int, now_ms: int) -> int:
        if self.active_zone is None:
            self.active_zone = raw_zone
            self.pending_zone = None
            self.pending_since_ms = 0
            return self.active_zone

        if raw_zone == self.active_zone:
            self.pending_zone = None
            self.pending_since_ms = 0
            return self.active_zone

        if self.pending_zone != raw_zone:
            self.pending_zone = raw_zone
            self.pending_since_ms = now_ms
            return self.active_zone

        if (now_ms - self.pending_since_ms) >= ZONE_HOLD_MS:
            self.active_zone = raw_zone
            self.pending_zone = None
            self.pending_since_ms = 0

        return self.active_zone

def main():
    outport, port_name, port_names = open_midi_out(None)
    if outport is None:
        print("No MIDI output ports found. Install/enable a synth or virtual MIDI output.")
        return
    print("Using MIDI out:", port_name)

    midi_all_notes_off(outport)
    midi_set_channel_volume(outport, CHANNEL_VOLUME_CC7)
    midi_set_expr(outport, DEFAULT_EXPR)
    midi_pitchwheel(outport, 0)

    cap = cv2.VideoCapture(CAM_INDEX, CAM_BACKEND)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, CAP_BUFFER)
    except Exception:
        pass

    grabber = FrameGrabber(cap)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    zone = ZoneDwell()
    active_notes: List[int] = []
    pending_off: Optional[Tuple[List[int], int]] = None  # (notes, off_at_ms)

    last_frame_t = time.time()
    fps = 0.0

    y_top = Y_TOP_DEFAULT
    y_bottom = Y_BOTTOM_DEFAULT

    spread_min: Optional[float] = None
    spread_max: Optional[float] = None

    fist = False
    releasing = False
    release_start_ms = 0
    release_from_expr = DEFAULT_EXPR
    last_expr_sent = DEFAULT_EXPR
    last_pitch_sent = 0

    while True:
        if DROP_OLD_FRAME_EACH_LOOP:
            grabber.get()

        frame_bgr = grabber.get()
        if frame_bgr is None:
            time.sleep(0.005)
            continue

        now = time.time()
        now_ms = int(now * 1000)

        dt = now - last_frame_t
        last_frame_t = now
        if dt > 1e-6:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        h, w = frame_bgr.shape[:2]

        infer_bgr = frame_bgr
        if INFER_DOWNSCALE != 1.0:
            infer_bgr = cv2.resize(
                frame_bgr,
                (int(w * INFER_DOWNSCALE), int(h * INFER_DOWNSCALE)),
                interpolation=cv2.INTER_LINEAR
            )

        infer_rgb = cv2.cvtColor(infer_bgr, cv2.COLOR_BGR2RGB)
        res = hands.process(infer_rgb)

        have_hand = False
        raw_zone = 0
        expr = DEFAULT_EXPR
        spread_ratio = 0.0
        spread_norm = 0.0
        fist_ratio = 10.0

        if res.multi_hand_landmarks:
            have_hand = True
            lms = res.multi_hand_landmarks[0].landmark

            iw = infer_bgr.shape[1]
            ih = infer_bgr.shape[0]
            sx = w / float(iw)
            sy = h / float(ih)

            def lm_xy(i: int) -> Tuple[float, float]:
                return (lms[i].x * iw * sx, lms[i].y * ih * sy)

            palm_pts = [lm_xy(i) for i in (0, 5, 9, 13, 17)]
            cx = sum(p[0] for p in palm_pts) / len(palm_pts)
            cy = sum(p[1] for p in palm_pts) / len(palm_pts)
            cy_n = cy / h

            palm_width = dist2(lm_xy(5), lm_xy(17))
            palm_width = max(1e-6, palm_width)

            thumb = lm_xy(4)
            pinky = lm_xy(20)

            spread_raw = dist2(thumb, pinky)
            spread_ratio = float(spread_raw / palm_width)

            if spread_min is None or spread_max is None:
                spread_min = spread_ratio * 0.90
                spread_max = spread_ratio * 1.10
            else:
                if spread_ratio < spread_min:
                    spread_min = spread_ratio
                else:
                    spread_min = spread_min + (spread_ratio - spread_min) * 0.001

                if spread_ratio > spread_max:
                    spread_max = spread_ratio
                else:
                    spread_max = spread_max + (spread_ratio - spread_max) * 0.001

                if (spread_max - spread_min) < 0.10:
                    mid = 0.5 * (spread_max + spread_min)
                    spread_min = mid - 0.05
                    spread_max = mid + 0.05

            spread_norm = clamp01((spread_ratio - spread_min) / (spread_max - spread_min + 1e-6))

            if y_bottom < y_top:
                y_top, y_bottom = y_bottom, y_top
            denom = (y_bottom - y_top)
            denom = denom if abs(denom) > 1e-6 else 1e-6
            t_top = clamp01((y_bottom - cy_n) / denom)
            raw_zone = int(np.clip(t_top * NUM_ZONES, 0, NUM_ZONES - 1))
            zone.update(raw_zone, now_ms)

            expr = int(lerp(MIN_EXPR, MAX_EXPR, t_top))
            expr = max(MIN_EXPR, min(MAX_EXPR, expr))

            palm_center = (cx, cy)
            tips = [lm_xy(i) for i in (4, 8, 12, 16, 20)]
            mean_tip = sum(dist2(p, palm_center) for p in tips) / len(tips)
            fist_ratio = float(mean_tip / palm_width)

            if not fist:
                if fist_ratio <= FIST_RATIO_ON:
                    fist = True
            else:
                if fist_ratio >= FIST_RATIO_OFF:
                    fist = False

            # landmarks (light)
            for i in (0, 4, 5, 8, 9, 12, 13, 16, 17, 20):
                px, py = lm_xy(i)
                cv2.circle(frame_bgr, (int(px), int(py)), 3, (90, 200, 90), -1)

            # Spread line + rotated label
            tpx, tpy = int(thumb[0]), int(thumb[1])
            ppx, ppy = int(pinky[0]), int(pinky[1])
            cv2.line(frame_bgr, (tpx, tpy), (ppx, ppy), (255, 255, 255), 2, cv2.LINE_AA)
            mid = ((tpx + ppx) // 2, (tpy + ppy) // 2)
            ang = math.degrees(math.atan2(ppy - tpy, ppx - tpx))
            draw_rotated_text(frame_bgr, f"{spread_ratio:.2f} (n {spread_norm:.2f})", mid, ang)

        # ---- MIDI state machine ----
        if fist and (active_notes or not releasing):
            if not releasing and active_notes:
                releasing = True
                release_start_ms = now_ms
                release_from_expr = last_expr_sent

        if releasing:
            t = (now_ms - release_start_ms) / max(1.0, float(FIST_RELEASE_MS))
            t = clamp01(t)
            fade_expr = int(lerp(release_from_expr, 0, t))
            if abs(fade_expr - last_expr_sent) >= 1:
                midi_set_expr(outport, fade_expr)
                last_expr_sent = fade_expr
            if last_pitch_sent != 0:
                midi_pitchwheel(outport, 0)
                last_pitch_sent = 0
            if t >= 1.0:
                if active_notes:
                    chord_note_off(outport, active_notes)
                    active_notes = []
                releasing = False
        else:
            if have_hand and not fist:
                if abs(expr - last_expr_sent) >= 1:
                    midi_set_expr(outport, expr)
                    last_expr_sent = expr

                current_zone = zone.active_zone if zone.active_zone is not None else raw_zone
                new_notes = ZONE_CHORDS.get(int(current_zone), ZONE_CHORDS[0]).copy()

                if not active_notes:
                    chord_note_on(outport, new_notes, NOTE_VELOCITY)
                    active_notes = new_notes
                else:
                    if new_notes != active_notes:
                        to_on = [n for n in new_notes if n not in active_notes]
                        to_off = [n for n in active_notes if n not in new_notes]
                        if to_on:
                            chord_note_on(outport, to_on, NOTE_VELOCITY)
                        if to_off:
                            pending_off = (to_off, now_ms + int(CHORD_SLUR_MS))
                        active_notes = new_notes

                if pending_off is not None:
                    notes_to_off, off_at = pending_off
                    if now_ms >= off_at:
                        chord_note_off(outport, notes_to_off)
                        pending_off = None

                # vibrato from spread_norm
                s = float(spread_norm)
                if s <= SPREAD_START_N:
                    depth = 0
                elif s >= SPREAD_FULL_N:
                    depth = VIB_MAX_DEPTH
                else:
                    tt = (s - SPREAD_START_N) / max(1e-6, (SPREAD_FULL_N - SPREAD_START_N))
                    depth = int(tt * VIB_MAX_DEPTH)

                vib = int(depth * math.sin(2.0 * math.pi * VIB_RATE_HZ * now))
                if abs(vib - last_pitch_sent) > PITCHWHEEL_DEADBAND:
                    midi_pitchwheel(outport, vib)
                    last_pitch_sent = vib

        # ---- UI ----
        x0, y0 = 12, 18
        y = y0

        expr01 = (last_expr_sent - MIN_EXPR) / max(1.0, (MAX_EXPR - MIN_EXPR))
        y = draw_small_bar(frame_bgr, x0, y, "VOL (CC11)", expr01, (80, 200, 255))

        y = draw_small_bar(frame_bgr, x0, y, "SPREAD", spread_norm if have_hand else 0.0, (70, 255, 150))
        # draw threshold markers on spread bar
        sbx = x0
        sby = y - (BAR_H + BAR_PAD) + 4
        for tmark in (SPREAD_START_N, SPREAD_FULL_N):
            lx = sbx + int(BAR_W * clamp01(tmark))
            cv2.line(frame_bgr, (lx, sby-2), (lx, sby+BAR_H+2), (255,255,255), 1, cv2.LINE_AA)

        vib_depth = abs(last_pitch_sent) / max(1.0, float(VIB_MAX_DEPTH))
        y = draw_small_bar(frame_bgr, x0, y, "VIB", vib_depth, (255, 150, 80))

        y = draw_small_bar(frame_bgr, x0, y, "FIST", 1.0 if fist else 0.0, (255, 255, 80))

        xr = w - 260
        yr = 22
        yr = put_text_small(frame_bgr, xr, yr, f"FPS: {fps:.1f}")
        yr = put_text_small(frame_bgr, xr, yr, f"MIDI: {port_name}")
        yr = put_text_small(frame_bgr, xr, yr, f"Zone: {zone.active_zone} (raw {raw_zone})")
        yr = put_text_small(frame_bgr, xr, yr, f"Notes: {active_notes}")
        yr = put_text_small(frame_bgr, xr, yr, f"Spread ratio: {spread_ratio:.2f}")
        if have_hand:
            yr = put_text_small(frame_bgr, xr, yr, f"Fist ratio: {fist_ratio:.2f} ({'ON' if fist else 'OFF'})")

        cv2.putText(
            frame_bgr,
            "Keys: Q quit | P cycle MIDI port | O print MIDI ports | R all notes off",
            (12, h - 14),
            FONT, 0.42, (220,220,220), 1, cv2.LINE_AA
        )

        # resizable window scaling
        show = frame_bgr
        try:
            _, _, ww, hh = cv2.getWindowImageRect(WINDOW_NAME)
            if ww > 50 and hh > 50 and (ww != w or hh != h):
                show = cv2.resize(frame_bgr, (ww, hh), interpolation=cv2.INTER_LINEAR)
        except Exception:
            pass

        cv2.imshow(WINDOW_NAME, show)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord('r'):
            midi_all_notes_off(outport)
            active_notes = []
            pending_off = None
            midi_pitchwheel(outport, 0)
            last_pitch_sent = 0
            midi_set_expr(outport, DEFAULT_EXPR)
            last_expr_sent = DEFAULT_EXPR
        if key == ord('o'):
            print("MIDI outputs:", port_names)
        if key == ord('p'):
            if port_names:
                try:
                    idx = port_names.index(port_name) if port_name in port_names else 0
                    idx = (idx + 1) % len(port_names)
                    new_name = port_names[idx]
                    outport.close()
                    outport = mido.open_output(new_name)
                    port_name = new_name
                    print("Switched MIDI out:", port_name)

                    midi_all_notes_off(outport)
                    midi_set_channel_volume(outport, CHANNEL_VOLUME_CC7)
                    midi_set_expr(outport, DEFAULT_EXPR)
                    last_expr_sent = DEFAULT_EXPR
                    midi_pitchwheel(outport, 0)
                    last_pitch_sent = 0
                    active_notes = []
                    pending_off = None
                except Exception as e:
                    print("Failed to switch port:", e)

    # cleanup
    try:
        midi_all_notes_off(outport)
        outport.close()
    except Exception:
        pass
    hands.close()
    grabber.stop()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
