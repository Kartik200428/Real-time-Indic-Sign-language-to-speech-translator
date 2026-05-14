"""
realtime_translator.py — Single-word diagnostic tool for ISL inference.

Focuses entirely on raw, frame-by-frame prediction accuracy.
No sentence logic, no speech engine, no smoothing streak.

Controls:
    q — quit

Run:
    venv312\\Scripts\\python.exe realtime_translator.py
"""

from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision as mv
from tensorflow.keras.models import load_model

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT           = Path(__file__).parent
HOLISTIC_MODEL = str(ROOT / "holistic_landmarker.task")
KERAS_MODEL    = str(ROOT / "models" / "isl_model_solo.keras")
FEATURES_DIR   = ROOT / "data" / "extracted_features"

# ── Constants (must match personal_collector.py exactly) ──────────────────────
MAX_FRAMES   = 30
POSE_DIM     = 33 * 4    # 132

FACE_KEY_INDICES = [
    70, 63, 105, 66, 107, 55, 65, 52,        # left eyebrow
    300, 293, 334, 296, 336, 285, 295, 282,   # right eyebrow
    33, 133, 159, 145,                         # left eye
    362, 263, 386, 374,                        # right eye
    61, 37,   0, 267, 291, 321, 314,  17,     # outer mouth
    78, 81,  13, 311, 308, 317,  14,  87,     # inner mouth
]
FACE_DIM     = len(FACE_KEY_INDICES) * 3   # 120
HAND_DIM     = 21 * 3    # 63
KEYPOINT_DIM = POSE_DIM + FACE_DIM + HAND_DIM * 2  # 378

CONFIDENCE_THRESHOLD = 0.50   # lower for diagnostics — see everything the model thinks

# ── Colours (BGR) ─────────────────────────────────────────────────────────────
C_BLACK  = (0,   0,   0)
C_WHITE  = (255, 255, 255)
C_GRAY   = (140, 140, 140)
C_YELLOW = (0,   215, 255)
C_GREEN  = (0,   210, 0)
C_RED    = (60,  60,  220)
C_CYAN   = (255, 200, 0)
C_BLUE   = (220, 140, 0)


# ── Label loading ─────────────────────────────────────────────────────────────

def load_labels() -> list[str]:
    return sorted(p.name for p in FEATURES_DIR.iterdir() if p.is_dir())


# ── MediaPipe setup ───────────────────────────────────────────────────────────

def build_holistic() -> mv.HolisticLandmarker:
    options = mv.HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HOLISTIC_MODEL),
        running_mode=mv.RunningMode.IMAGE,
        min_face_detection_confidence=0.5,
        min_face_landmarks_confidence=0.5,
        min_pose_detection_confidence=0.5,
        min_pose_landmarks_confidence=0.5,
        min_hand_landmarks_confidence=0.5,
    )
    return mv.HolisticLandmarker.create_from_options(options)


# ── Wrist-relative hand normalization ─────────────────────────────────────────

def _wrist_relative(lm_list) -> np.ndarray:
    """
    Convert 21 hand landmarks to wrist-relative coordinates.
    Subtracting landmark-0 (wrist) from every landmark makes the gesture
    position- and distance-independent, reducing domain shift between
    dataset signers and the live camera.
    """
    coords = np.array([[lm.x, lm.y, lm.z] for lm in lm_list], dtype=np.float32)
    coords -= coords[0]   # wrist becomes (0, 0, 0); all others are relative to it
    return coords.flatten()


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_keypoints(result: mv.HolisticLandmarkerResult) -> np.ndarray:
    """
    Build the 1692-float feature vector.
    Hands use wrist-relative normalization.
    Pose and face are left as absolute (they provide body context).
    """
    # Pose — absolute (132)
    if result.pose_landmarks:
        pose = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility] for lm in result.pose_landmarks],
            dtype=np.float32,
        ).flatten()
    else:
        pose = np.zeros(POSE_DIM, dtype=np.float32)

    # Face — 40 key points, nose-relative (120)
    if result.face_landmarks and len(result.face_landmarks) > max(FACE_KEY_INDICES):
        lms  = result.face_landmarks
        nose = np.array([lms[1].x, lms[1].y, lms[1].z], dtype=np.float32)
        pts  = np.array([[lms[i].x, lms[i].y, lms[i].z]
                          for i in FACE_KEY_INDICES], dtype=np.float32)
        pts -= nose
        face = pts.flatten()
    else:
        face = np.zeros(FACE_DIM, dtype=np.float32)

    # Left hand — wrist-relative (63)
    lh = _wrist_relative(result.left_hand_landmarks) \
        if result.left_hand_landmarks else np.zeros(HAND_DIM, dtype=np.float32)

    # Right hand — wrist-relative (63)
    rh = _wrist_relative(result.right_hand_landmarks) \
        if result.right_hand_landmarks else np.zeros(HAND_DIM, dtype=np.float32)

    return np.concatenate([pose, face, lh, rh])


# ── UI ────────────────────────────────────────────────────────────────────────

def draw_ui(
    frame: np.ndarray,
    hands_present: bool,
    buffer_pct: float,           # 0.0 – 1.0, how full the 30-frame buffer is
    top3: list[tuple[str, float]],
) -> None:
    h, w = frame.shape[:2]

    # ── Background panels ─────────────────────────────────────────────────
    def panel(y0, y1, alpha=0.6):
        ov = frame.copy()
        cv2.rectangle(ov, (0, y0), (w, y1), C_BLACK, -1)
        cv2.addWeighted(ov, alpha, frame, 1 - alpha, 0, frame)

    panel(0, 60)           # top bar
    panel(h - 40, h, 0.5)  # bottom hint bar

    # ── Gate status / current word ────────────────────────────────────────
    if not hands_present:
        cv2.putText(frame, "Waiting for sign ...",
                    (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, C_GRAY, 2)
    elif not top3:
        cv2.putText(frame, "Buffering ...",
                    (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, C_GRAY, 2)
    else:
        word, conf = top3[0]
        colour = C_GREEN if conf >= CONFIDENCE_THRESHOLD else C_RED
        cv2.putText(frame, f"{word}",
                    (12, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.2, colour, 3)
        cv2.putText(frame, f"{conf * 100:.1f}%",
                    (w - 110, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 2)

    # ── Buffer fill bar ───────────────────────────────────────────────────
    bar_full_w = w - 20
    filled     = int(bar_full_w * buffer_pct)
    cv2.rectangle(frame, (10, 55), (10 + bar_full_w, 62), C_GRAY, 1)
    cv2.rectangle(frame, (10, 55), (10 + filled, 62),
                  C_GREEN if buffer_pct >= 1.0 else C_BLUE, -1)

    # ── Top-3 debug panel ─────────────────────────────────────────────────
    if top3:
        box_w, box_h = 260, 18 + len(top3) * 30
        bx, by = w - box_w - 10, 70
        ov2 = frame.copy()
        cv2.rectangle(ov2, (bx - 6, by - 6), (bx + box_w, by + box_h), C_BLACK, -1)
        cv2.addWeighted(ov2, 0.65, frame, 0.35, 0, frame)
        cv2.putText(frame, "TOP 3  (raw / no smoothing)",
                    (bx, by + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_CYAN, 1)

        for i, (word, prob) in enumerate(top3):
            y    = by + 38 + i * 30
            pct  = prob * 100
            bar  = int(pct / 100 * (box_w - 10))
            b_colour = C_GREEN if i == 0 and prob >= CONFIDENCE_THRESHOLD else C_BLUE
            cv2.rectangle(frame, (bx, y - 14), (bx + bar, y), b_colour, -1)
            cv2.rectangle(frame, (bx, y - 14), (bx + box_w - 10, y), C_GRAY, 1)
            label_colour = C_YELLOW if i == 0 else C_WHITE
            cv2.putText(frame, f"{word:<20}{pct:5.1f}%",
                        (bx + 4, y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, label_colour, 1)

    # ── Bottom hint ───────────────────────────────────────────────────────
    cv2.putText(frame, "Diagnostic mode — raw frame-by-frame output   [Q] quit",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_GRAY, 1)


# ── Main loop ─────────────────────────────────────────────────────────────────

def run(camera_index: int = 0) -> None:
    print("Loading labels ...")
    labels = load_labels()
    print(f"  {len(labels)} classes loaded.\n")

    print("Loading Keras model ...")
    model = load_model(KERAS_MODEL)
    print("  Model loaded.\n")

    sequence: deque[np.ndarray] = deque(maxlen=MAX_FRAMES)
    top3: list[tuple[str, float]] = []

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_index}")
    print("Camera open.  Press Q to quit.\n")

    with build_holistic() as holistic:
        while True:
            ok, raw_frame = cap.read()
            if not ok:
                continue

            # ── Mirror fix ────────────────────────────────────────────────
            # Flip horizontally before MediaPipe so your right hand is
            # detected as right_hand_landmarks (matching training data).
            # The same flipped frame is shown to you — looks like a normal
            # face-on recording, not a selfie mirror.
            frame = cv2.flip(raw_frame, 1)

            # ── Landmark extraction ───────────────────────────────────────
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result   = holistic.detect(mp_image)

            # ── Hand gate ─────────────────────────────────────────────────
            hands_present = bool(
                result.left_hand_landmarks or result.right_hand_landmarks
            )

            if not hands_present:
                sequence.clear()
                top3 = []
                draw_ui(frame, False, 0.0, [])
                cv2.imshow("ISL Diagnostic", frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
                continue

            # ── Build buffer ──────────────────────────────────────────────
            sequence.append(extract_keypoints(result))
            buffer_pct = len(sequence) / MAX_FRAMES

            # ── Raw frame-by-frame prediction (no streak required) ────────
            if len(sequence) == MAX_FRAMES:
                X     = np.expand_dims(np.array(sequence, dtype=np.float32), axis=0)
                probs = model.predict(X, verbose=0)[0]
                top_i = np.argsort(probs)[::-1][:3]
                top3  = [(labels[i], float(probs[i])) for i in top_i]

            # ── Draw & show ───────────────────────────────────────────────
            draw_ui(frame, hands_present, buffer_pct, top3)
            cv2.imshow("ISL Diagnostic", frame)

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
