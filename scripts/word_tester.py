"""
word_tester.py — Test all 50 words one by one.

Left panel  : reference video loop so you can see the correct sign.
Right panel : live webcam with real-time prediction.

Controls:
    SPACE  — next word (mark as OK)
    F      — flag this word as FAILING and move on
    Q      — quit early

At the end a summary shows every flagged word.

Run:
    venv312\\Scripts\\python.exe word_tester.py
"""

import re
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision as mv
from tensorflow.keras.models import load_model

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT           = Path(__file__).parent.parent
RAW_DIR        = ROOT / "data" / "raw"
HOLISTIC_MODEL = str(ROOT / "mediapipe_models" / "holistic_landmarker.task")
KERAS_MODEL    = str(ROOT / "models" / "isl_model_solo.keras")
FEATURES_DIR   = ROOT / "data" / "extracted_features"

# ── Feature constants (must match personal_collector.py) ─────────────────────
MAX_FRAMES = 30
POSE_DIM   = 33 * 4

FACE_KEY_INDICES = [
    70, 63, 105, 66, 107, 55, 65, 52,
    300, 293, 334, 296, 336, 285, 295, 282,
    33, 133, 159, 145,
    362, 263, 386, 374,
    61, 37,   0, 267, 291, 321, 314,  17,
    78, 81,  13, 311, 308, 317,  14,  87,
]
FACE_DIM     = len(FACE_KEY_INDICES) * 3
HAND_DIM     = 21 * 3
KEYPOINT_DIM = POSE_DIM + FACE_DIM + HAND_DIM * 2   # 378

CONFIDENCE_THRESHOLD = 0.50

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".MOV", ".MP4"}

TARGET_WORDS = [
    "I", "you", "he", "she", "they", "we", "it",
    "Mother", "Father", "Brother", "Sister", "Friend", "Family", "Man", "Woman",
    "Teacher", "Doctor", "Police", "Student",
    "Exercise", "Sign", "Dream", "Sport",
    "good", "bad", "happy", "sad", "sick", "healthy",
    "strong", "weak", "young", "old", "Deaf", "alive",
    "Today", "Tomorrow", "Yesterday", "Morning", "Night", "Time",
    "How are you", "Hello", "Thank you", "Good Morning",
    "Hospital", "School", "House", "Medicine", "Patient",
]

# ── Colours (BGR) ─────────────────────────────────────────────────────────────
C_BLACK  = (0,   0,   0)
C_WHITE  = (255, 255, 255)
C_GRAY   = (140, 140, 140)
C_GREEN  = (0,   210, 0)
C_RED    = (60,  60,  220)
C_YELLOW = (0,   215, 255)
C_ORANGE = (0,   140, 255)
C_CYAN   = (255, 200, 0)


# ── Raw video scanner ─────────────────────────────────────────────────────────

def _normalize(s):
    return re.sub(r"[\s_\-]", "", s).lower()

def _word_from_folder(name):
    m = re.match(r"^\d+\.\s*(.+)$", name)
    return m.group(1).strip() if m else name.strip()

def scan_raw_videos():
    folder_map = {}
    for cat in RAW_DIR.iterdir():
        if not cat.is_dir():
            continue
        for wdir in cat.iterdir():
            if not wdir.is_dir():
                continue
            display = _word_from_folder(wdir.name)
            key = _normalize(display)
            if key in folder_map:
                continue
            videos = [f for f in wdir.iterdir() if f.suffix in VIDEO_EXTS]
            if videos:
                folder_map[key] = sorted(videos)[0]
    result = {}
    for word in TARGET_WORDS:
        key = _normalize(word)
        if key in folder_map:
            result[word] = folder_map[key]
    return result


# ── MediaPipe ─────────────────────────────────────────────────────────────────

def build_holistic():
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


# ── Feature extraction ────────────────────────────────────────────────────────

def _wrist_relative(lm_list):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in lm_list], dtype=np.float32)
    coords -= coords[0]
    return coords.flatten()

def extract_keypoints(result):
    pose = np.array(
        [[lm.x, lm.y, lm.z, lm.visibility] for lm in result.pose_landmarks],
        dtype=np.float32,
    ).flatten() if result.pose_landmarks else np.zeros(POSE_DIM, dtype=np.float32)

    if result.face_landmarks and len(result.face_landmarks) > max(FACE_KEY_INDICES):
        lms  = result.face_landmarks
        nose = np.array([lms[1].x, lms[1].y, lms[1].z], dtype=np.float32)
        pts  = np.array([[lms[i].x, lms[i].y, lms[i].z]
                          for i in FACE_KEY_INDICES], dtype=np.float32)
        pts -= nose
        face = pts.flatten()
    else:
        face = np.zeros(FACE_DIM, dtype=np.float32)

    lh = _wrist_relative(result.left_hand_landmarks) \
        if result.left_hand_landmarks else np.zeros(HAND_DIM, dtype=np.float32)
    rh = _wrist_relative(result.right_hand_landmarks) \
        if result.right_hand_landmarks else np.zeros(HAND_DIM, dtype=np.float32)

    return np.concatenate([pose, face, lh, rh])


# ── Frame helpers ─────────────────────────────────────────────────────────────

PANEL_W, PANEL_H = 640, 480

def _resize(frame):
    return cv2.resize(frame, (PANEL_W, PANEL_H))

def _overlay_text(frame, text, y, colour, scale=0.9, thickness=2, center=False):
    if center:
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        x = (frame.shape[1] - tw) // 2
    else:
        x = 12
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thickness)

def _dark_bar(frame, y0, y1, alpha=0.65):
    ov = frame.copy()
    cv2.rectangle(ov, (0, y0), (frame.shape[1], y1), C_BLACK, -1)
    cv2.addWeighted(ov, alpha, frame, 1 - alpha, 0, frame)


# ── Main ──────────────────────────────────────────────────────────────────────

def run(camera_index=0):
    print("Scanning reference videos ...")
    video_map = scan_raw_videos()
    print(f"  Found {len(video_map)}/{len(TARGET_WORDS)} reference videos.\n")

    print("Loading label map ...")
    labels = sorted(p.name for p in FEATURES_DIR.iterdir() if p.is_dir())
    print(f"  {len(labels)} classes.\n")

    print("Loading model ...")
    model = load_model(KERAS_MODEL)
    print("  Done.\n")

    cam = cv2.VideoCapture(camera_index)
    if not cam.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_index}")

    win = "ISL Word Tester"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, PANEL_W * 2, PANEL_H)

    failed_words = []
    sequence = deque(maxlen=MAX_FRAMES)
    prediction = ("", 0.0)

    with build_holistic() as holistic:
        for word_idx, word in enumerate(TARGET_WORDS):
            sequence.clear()
            prediction = ("", 0.0)

            # open reference video (or None)
            ref_cap = None
            if word in video_map:
                ref_cap = cv2.VideoCapture(str(video_map[word]))
                if not ref_cap.isOpened():
                    ref_cap = None

            action = None
            while action is None:
                # ── reference frame ───────────────────────────────────
                if ref_cap is not None:
                    ok, ref_frame = ref_cap.read()
                    if not ok:
                        ref_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ok, ref_frame = ref_cap.read()
                    ref_panel = _resize(ref_frame) if ok else np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
                else:
                    ref_panel = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
                    _overlay_text(ref_panel, "No reference video", PANEL_H // 2,
                                  C_GRAY, center=True)

                # ── webcam frame + inference ───────────────────────────
                ok, raw = cam.read()
                if not ok:
                    cam_panel = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
                else:
                    frame = cv2.flip(raw, 1)
                    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    result = holistic.detect(mp_img)

                    hands_present = bool(
                        result.left_hand_landmarks or result.right_hand_landmarks
                    )
                    if hands_present:
                        sequence.append(extract_keypoints(result))
                        if len(sequence) == MAX_FRAMES:
                            X     = np.expand_dims(np.array(sequence, dtype=np.float32), 0)
                            probs = model.predict(X, verbose=0)[0]
                            top_i = int(np.argmax(probs))
                            prediction = (labels[top_i], float(probs[top_i]))
                    else:
                        sequence.clear()
                        prediction = ("", 0.0)

                    cam_panel = _resize(frame)

                # ── draw reference panel labels ───────────────────────
                _dark_bar(ref_panel, 0, 52)
                _overlay_text(ref_panel, f"REFERENCE  ({word_idx+1}/{len(TARGET_WORDS)})",
                              18, C_GRAY, scale=0.5)
                _overlay_text(ref_panel, word, 46, C_YELLOW, scale=1.0)

                _dark_bar(ref_panel, PANEL_H - 36, PANEL_H, 0.55)
                _overlay_text(ref_panel, "SPACE = OK   F = fail   Q = quit",
                              PANEL_H - 12, C_WHITE, scale=0.5)

                # ── draw webcam panel labels ───────────────────────────
                _dark_bar(cam_panel, 0, 52)
                _overlay_text(cam_panel, "YOUR WEBCAM", 18, C_GRAY, scale=0.5)

                pred_word, conf = prediction
                if pred_word:
                    correct = pred_word.lower() == word.lower()
                    colour  = C_GREEN if correct else C_RED
                    _overlay_text(cam_panel, pred_word, 46, colour, scale=1.0)
                    cv2.putText(cam_panel, f"{conf*100:.1f}%",
                                (PANEL_W - 100, 46),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)
                else:
                    _overlay_text(cam_panel, "Waiting for hands...", 46, C_GRAY, scale=0.7)

                # buffer bar
                buf_pct = len(sequence) / MAX_FRAMES
                bw = PANEL_W - 20
                cv2.rectangle(cam_panel, (10, 55), (10 + bw, 62), C_GRAY, 1)
                cv2.rectangle(cam_panel, (10, 55),
                              (10 + int(bw * buf_pct), 62),
                              C_GREEN if buf_pct >= 1.0 else C_CYAN, -1)

                # ── combine and show ──────────────────────────────────
                combined = np.hstack([ref_panel, cam_panel])
                cv2.imshow(win, combined)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    action = "quit"
                elif key == ord(" "):
                    action = "ok"
                elif key == ord("f"):
                    action = "fail"
                    failed_words.append(word)
                    print(f"  FLAGGED: {word}")

            if ref_cap is not None:
                ref_cap.release()

            if action == "quit":
                break

    cam.release()
    cv2.destroyAllWindows()

    print(f"\n{'='*50}")
    print(f"Words tested  : {word_idx + 1}")
    print(f"Flagged fails : {len(failed_words)}")
    if failed_words:
        print(f"\nFailing words:")
        for w in failed_words:
            print(f"  - {w}")
    else:
        print("All words passed!")


if __name__ == "__main__":
    run()
