import math
import sys
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# =============================== SETTINGS ==================================
# Training
RFDETR_SIZE = "medium"          # small | base | medium | large (must match the checkpoint)
RESOLUTION  = 736               # RF-DETR input resolution
SLICE_SIZE  = 736               # SAHI tile size
OVERLAP     = 0.2               # SAHI tile overlap
TRACK_CONF   = 0.3              # low threshold: keep faint balls for tracking
ACQUIRE_CONF = 0.7              # high threshold: picking the ball at address


ROI_POLYGON   = [(0.15, 0.5), (0.85, 0.5), (0.85, 1.0), (0.15, 1.0)]
TARGET_ANCHOR = (0.20, 0.90)    # "bottom_left"

# Tracking
STABLE_FRAMES     = 6           # consecutive stable detections before we "arm"
STABLE_RADIUS_F   = 1.5         # max move (ball radii) to count as "still"
IMPACT_MOVE_F     = 3.0         # move from origin (ball radii) that means "struck"
IMPACT_ABSENT     = 8           # frames with no ball found -> re-acquire
MAX_LAUNCH_JUMP_F = 0.6         # max launch jump (fraction of max frame dim)
MAX_PREDICT       = 20          # frames to extrapolate after detection is lost
LAUNCH_V_AT       = 3           # measured frame at which launch velocity is sampled
BALL_DIAMETER_M   = 0.04267     # regulation golf ball, for px -> metres scale
MEAS_NOISE, PROC_NOISE_POS, PROC_NOISE_VEL = 1.0, 1e-2, 5e-1

STATIC_MATCH_F, STATIC_MIN_HITS, STATIC_EXCLUDE_F = 2.0, 5, 3.0
CONFIRM_MIN_STEP_F, CONFIRM_MAX_MISS = 1.0, 1
GATE_MIN_R_F, GATE_V_MULT, VEL_CONSIST_W = 6.0, 3.0, 0.5
STATIC_TRACK_REJECT = 3


# ============================== MODEL ======================================
def load_model(checkpoint, device):
    """SAHI-wrapped detector. `.pt` -> YOLO (ultralytics), otherwise RF-DETR."""
    checkpoint = str(checkpoint)
    if checkpoint.endswith(".pt"):
        return AutoDetectionModel.from_pretrained(
            model_type="ultralytics", model_path=checkpoint,
            confidence_threshold=TRACK_CONF, device=device)

    from rfdetr import RFDETRSmall, RFDETRBase, RFDETRMedium, RFDETRLarge
    cls = {"small": RFDETRSmall, "base": RFDETRBase,
           "medium": RFDETRMedium, "large": RFDETRLarge}[RFDETR_SIZE]
    # inference only: `pretrain_weights` is how rfdetr loads a trained checkpoint
    rf = cls(resolution=RESOLUTION, pretrain_weights=checkpoint)
    try:
        rf.optimize_for_inference()
    except Exception as e:
        print(f"[warn] optimize_for_inference() failed, running unoptimized: {e}")
    return AutoDetectionModel.from_pretrained(
        model_type="roboflow", model=rf,
        confidence_threshold=TRACK_CONF, device=device)


def default_device():
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def detect_in_window(frame_bgr, model, win, W, H):
    """Sliced detection inside `win` = (x0, y0, x1, y1).
    Returns [(x0, y0, x1, y1, score), ...] in full-frame coordinates."""
    x0, y0 = max(0, int(win[0])), max(0, int(win[1]))
    x1, y1 = min(W, int(win[2])), min(H, int(win[3]))
    if x1 - x0 < 16 or y1 - y0 < 16:
        return []
    rgb = cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
    result = get_sliced_prediction(
        rgb, model, slice_height=SLICE_SIZE, slice_width=SLICE_SIZE,
        overlap_height_ratio=OVERLAP, overlap_width_ratio=OVERLAP, verbose=0)
    dets = []
    for op in result.object_prediction_list:
        bx0, by0, bx1, by1 = op.bbox.to_xyxy()
        dets.append((bx0 + x0, by0 + y0, bx1 + x0, by1 + y0, op.score.value))
    return dets


def center(det):
    return (det[0] + det[2]) / 2.0, (det[1] + det[3]) / 2.0


# =============================== ROI =======================================
def point_in_polygon(x, y, poly):
    inside, j = False, len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def select_target(dets, roi_px, anchor_px):
    """High-confidence detection inside the ROI closest to the anchor, or None."""
    cands = [d for d in dets if point_in_polygon(*center(d), roi_px)]
    if not cands:
        return None
    ax, ay = anchor_px
    return min(cands, key=lambda d: (center(d)[0] - ax) ** 2 + (center(d)[1] - ay) ** 2)


# ============================ KALMAN FILTER ================================
class KalmanFilter:
    """Constant-velocity model with gravity as a control input."""

    def __init__(self, gravity_pxf2):
        kf = cv2.KalmanFilter(4, 2, 1)
        kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1],
                                        [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        kf.controlMatrix = np.array([[0.0], [0.5], [0.0], [1.0]], np.float32)
        kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kf.processNoiseCov = np.diag([PROC_NOISE_POS, PROC_NOISE_POS,
                                      PROC_NOISE_VEL, PROC_NOISE_VEL]).astype(np.float32)
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * MEAS_NOISE
        kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.kf = kf
        self.g = np.array([[gravity_pxf2]], np.float32)

    def init(self, x, y, vx, vy):
        self.kf.statePost = np.array([[x], [y], [vx], [vy]], np.float32)
        self.kf.statePre = self.kf.statePost.copy()

    def predict(self):
        s = self.kf.predict(self.g)
        return float(s[0][0]), float(s[1][0]), float(s[2][0]), float(s[3][0])

    def correct(self, x, y):
        self.kf.correct(np.array([[x], [y]], np.float32))

    def velocity(self):
        s = self.kf.statePost
        return float(s[2][0]), float(s[3][0])


# ============================== TRACKER ====================================
class State(Enum):
    WAITING = "waiting"          # looking for a ball at rest in the ROI
    ARMED = "armed"              # ball at rest found, waiting for impact
    TRACKING = "tracking"        # ball in flight, measured this frame
    PREDICTING = "predicting"    # ball in flight, extrapolated this frame
    DONE = "done"


class Tracker:
    """State machine + Kalman tracker for a single launched golf ball."""

    def __init__(self, fps, W, H):
        self.fps, self.W, self.H = fps, W, H
        self.state = State.WAITING
        self.recent = []
        self.origin = None
        self.ball_r = 10.0
        self.kf = None
        self.t0 = None
        self.predict_count = 0
        self.absent = 0
        self.measured_count = 0
        self.launch_v = None
        self.path = []                 # [(t, x, y, measured_bool)]
        self.static_map = []           # stationary decoys: [x, y, hits, last_frame]
        self.pending = None            # (frame_idx, x, y) launch awaiting confirmation
        self.pending_miss = 0
        self.last_meas = None
        self.static_meas_count = 0

    # ---------- helpers ----------
    def _px_per_m(self):
        return 2 * self.ball_r / BALL_DIAMETER_M

    def _near_origin(self, cx, cy):
        return math.hypot(cx - self.origin[0], cy - self.origin[1]) <= IMPACT_MOVE_F * self.ball_r

    def _origin_occupied(self, dets):
        return self.origin is not None and any(self._near_origin(*center(d)) for d in dets)

    def _update_static_map(self, dets, frame_idx):
        match_r = STATIC_MATCH_F * self.ball_r
        for d in dets:
            cx, cy = center(d)
            for e in self.static_map:
                if math.hypot(cx - e[0], cy - e[1]) <= match_r:
                    e[0] = 0.8 * e[0] + 0.2 * cx
                    e[1] = 0.8 * e[1] + 0.2 * cy
                    e[2] += 1
                    e[3] = frame_idx
                    break
            else:
                self.static_map.append([cx, cy, 1, frame_idx])

    def _is_static_decoy(self, cx, cy):
        excl_r = STATIC_EXCLUDE_F * self.ball_r
        for x, y, hits, _ in self.static_map:
            if hits < STATIC_MIN_HITS:
                continue
            if self.origin is not None and \
               math.hypot(x - self.origin[0], y - self.origin[1]) <= excl_r:
                continue           # never blacklist the tee itself
            if math.hypot(cx - x, cy - y) <= excl_r:
                return True
        return False

    # ---------- acquisition ----------
    def _arm_update(self, det, frame_idx):
        cx, cy = center(det)
        self.ball_r = max(4.0, (det[2] - det[0] + det[3] - det[1]) / 4.0)
        self.recent = (self.recent + [(cx, cy)])[-STABLE_FRAMES:]
        if len(self.recent) < STABLE_FRAMES:
            return
        xs = [p[0] for p in self.recent]
        ys = [p[1] for p in self.recent]
        if math.hypot(max(xs) - min(xs), max(ys) - min(ys)) <= STABLE_RADIUS_F * self.ball_r:
            self.origin = (sum(xs) / len(xs), sum(ys) / len(ys))
            self.state = State.ARMED

    def _find_departed(self, dets):
        """A detection that has just left the origin (not a static decoy)."""
        ox, oy = self.origin
        lo = IMPACT_MOVE_F * self.ball_r
        hi = MAX_LAUNCH_JUMP_F * max(self.W, self.H)
        best, best_d = None, 1e18
        for d in dets:
            cx, cy = center(d)
            if self._is_static_decoy(cx, cy):
                continue
            dist = math.hypot(cx - ox, cy - oy)
            if lo < dist <= hi and cy <= oy + 2 * self.ball_r and dist < best_d:
                best, best_d = (cx, cy), dist
        return best

    def _confirm_launch(self, frame_idx, dets):
        """The pending candidate must still be moving away from the tee."""
        ox, oy = self.origin
        pf, px, py = self.pending
        min_step = CONFIRM_MIN_STEP_F * self.ball_r
        d_pending = math.hypot(px - ox, py - oy)
        best, best_step = None, 1e18
        for d in dets:
            cx, cy = center(d)
            if self._is_static_decoy(cx, cy):
                continue
            step = math.hypot(cx - px, cy - py)
            if math.hypot(cx - ox, cy - oy) >= d_pending + min_step \
               and step >= min_step and cy <= oy + 2 * self.ball_r and step < best_step:
                best, best_step = (cx, cy), step
        if best is None:
            self.pending_miss += 1
            if self.pending_miss > CONFIRM_MAX_MISS:
                self.pending, self.pending_miss = None, 0
            return False
        k = max(1, frame_idx - pf)
        self._launch(frame_idx, best, (best[0] - px) / k, (best[1] - py) / k)
        return True

    def _launch(self, frame_idx, pos, vx, vy):
        gravity = 9.81 * self._px_per_m() / self.fps ** 2      # px / frame^2
        self.kf = KalmanFilter(gravity)
        self.kf.init(pos[0], pos[1], vx, vy)
        self.t0 = frame_idx
        self.path = [(0, pos[0], pos[1], True)]
        self.measured_count = 1
        self.predict_count = self.absent = self.pending_miss = self.static_meas_count = 0
        self.pending = None
        self.last_meas = pos
        self.state = State.TRACKING

    # ---------- main step ----------
    def update(self, frame_idx, acquire_target, dets):
        if self.state == State.WAITING:
            self._update_static_map(dets, frame_idx)
            if acquire_target is not None:
                self._arm_update(acquire_target, frame_idx)

        elif self.state == State.ARMED:
            self._update_static_map(dets, frame_idx)
            # ball still resting on the tee -> nothing launched yet
            if acquire_target is not None and self._near_origin(*center(acquire_target)):
                self._arm_update(acquire_target, frame_idx)
                self.absent, self.pending = 0, None
            elif self._origin_occupied(dets):
                self.absent, self.pending = 0, None
            else:
                # tee empty: confirm a pending launch, or start one
                if self.pending is not None:
                    if self._confirm_launch(frame_idx, dets):
                        return
                else:
                    departed = self._find_departed(dets)
                    if departed is not None:
                        self.pending, self.pending_miss = (frame_idx, *departed), 0
                        return
                self.absent += 1
                if self.absent > IMPACT_ABSENT:            # ball gone: re-acquire
                    self.state = State.WAITING
                    self.recent, self.origin, self.pending = [], None, None

        elif self.state in (State.TRACKING, State.PREDICTING):
            self._step(frame_idx, dets)

    def _step(self, frame_idx, dets):
        px, py, vx, vy = self.kf.predict()                 # ballistic prediction
        gate = max(GATE_MIN_R_F * self.ball_r, GATE_V_MULT * math.hypot(vx, vy))

        # associate: distance to prediction + velocity-consistency penalty
        lx, ly = self.path[-1][1], self.path[-1][2]
        chosen, best_score = None, 1e18
        for d in dets:
            cx, cy = center(d)
            if self._is_static_decoy(cx, cy):
                continue
            dist = math.hypot(cx - px, cy - py)
            if dist > gate:
                continue
            score = dist + VEL_CONSIST_W * math.hypot(cx - lx - vx, cy - ly - vy)
            if score < best_score:
                best_score, chosen = score, (cx, cy)

        # a "measurement" that never moves is scenery, not a flying ball
        if chosen is not None and self.last_meas is not None:
            if math.hypot(chosen[0] - self.last_meas[0], chosen[1] - self.last_meas[1]) < self.ball_r:
                self.static_meas_count += 1
                if self.static_meas_count >= STATIC_TRACK_REJECT:
                    self.static_map.append([chosen[0], chosen[1], 999, frame_idx])
                    self.static_meas_count = 0
                    chosen = None
            else:
                self.static_meas_count = 0

        if chosen is not None:
            self.kf.correct(*chosen)
            self.last_meas = chosen
            px, py = chosen
            self.state = State.TRACKING
            self.predict_count = 0
            self.measured_count += 1
            if self.measured_count == LAUNCH_V_AT:
                self.launch_v = self.kf.velocity()
        else:
            self.state = State.PREDICTING
            self.predict_count += 1

        self.path.append((frame_idx - self.t0, px, py, chosen is not None))
        off_screen = px < -0.2 * self.W or px > 1.2 * self.W or py > self.H
        if self.predict_count > MAX_PREDICT or off_screen:
            self.state = State.DONE

    def search_window(self, roi_box):
        """Crop (x0, y0, x1, y1) where the ball can be on the next frame."""
        if self.state == State.ARMED:
            half = MAX_LAUNCH_JUMP_F * max(self.W, self.H)
            ox, oy = self.origin
            return (ox - half, oy - half, ox + half, oy + half)
        if self.state in (State.TRACKING, State.PREDICTING):
            vx, vy = self.kf.velocity()
            nx, ny = self.path[-1][1] + vx, self.path[-1][2] + vy
            half = max(GATE_MIN_R_F * self.ball_r,
                       GATE_V_MULT * math.hypot(vx, vy)) + 4 * self.ball_r
            return (nx - half, ny - half, nx + half, ny + half)
        return roi_box

    def metrics(self):
        if self.kf is None or self.measured_count < 2:
            return None
        vx, vy = self.launch_v or self.kf.velocity()
        speed_ms = math.hypot(vx, vy) * self.fps / self._px_per_m()
        return {"launch_angle_deg": round(math.degrees(math.atan2(-vy, vx)), 1),
                "ball_speed_ms": round(speed_ms, 1),
                "ball_speed_mph": round(speed_ms * 2.237, 1),
                "measured_frames": self.measured_count}


# ============================== PIPELINE ===================================
def track_video(input_path, output_path, checkpoint, model=None, device=None):
    """Track the ball in `input_path` and save a video with its trajectory
    drawn (green = measured, yellow = predicted) to `output_path`.
    Returns the launch metrics dict, or None if the shot wasn't tracked.

    Pass a pre-built `model` (from load_model) to avoid reloading it when
    processing many videos.
    """
    if model is None:
        model = load_model(checkpoint, device or default_device())

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise IOError(f"cannot open {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    roi_px = [(x * W, y * H) for x, y in ROI_POLYGON]
    margin = 0.05 * max(W, H)
    roi_box = (min(x for x, _ in roi_px) - margin, min(y for _, y in roi_px) - margin,
               max(x for x, _ in roi_px) + margin, max(y for _, y in roi_px) + margin)
    anchor_px = (TARGET_ANCHOR[0] * W, TARGET_ANCHOR[1] * H)
    tracker = Tracker(fps=fps, W=W, H=H)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # ---- detect + track, only where the ball can possibly be ----
        if tracker.state != State.DONE:
            dets = detect_in_window(frame, model, tracker.search_window(roi_box), W, H)
            acquire_target = None
            if tracker.state in (State.WAITING, State.ARMED):
                hi = [d for d in dets if d[4] >= ACQUIRE_CONF]
                acquire_target = select_target(hi, roi_px, anchor_px)
            tracker.update(frame_idx, acquire_target, dets)

        # ---- draw trajectory ----
        for (_, x0, y0, _), (_, x1, y1, measured) in zip(tracker.path, tracker.path[1:]):
            col = (0, 255, 0) if measured else (0, 255, 255)
            cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), col, 2)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    return tracker.metrics()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("usage: python track_video.py <input_video> <output_video> <checkpoint>")
    metrics = track_video(sys.argv[1], sys.argv[2], sys.argv[3])
    print(f"Saved {sys.argv[2]}  metrics={metrics}")
