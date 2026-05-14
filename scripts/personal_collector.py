"""
personal_collector.py — Record a personalised ISL dataset for fine-tuning.

For each of the 50 target words this script:
  1. Plays the first raw reference video on a loop so you can learn the sign.
  2. Opens your webcam and records 10 × 30-frame landmark sequences.
  3. Applies wrist-relative normalisation and saves (30, 1692) .npy arrays
     directly to  data/personal_features/<Word>/seq_NNNN.npy

Controls (during tutorial):
    C — ready to record
    S — skip this word entirely
    Q — quit the whole collector

Controls (during webcam / recording):
    R — start recording the 10 sequences for the current word
    S — skip this word and move to the next
    Q — quit

Run:
    venv312\\Scripts\\python.exe personal_collector.py
"""

import re
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision as mv

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).parent
RAW_DIR      = ROOT / "data" / "raw"
OUT_DIR      = ROOT / "data" / "personal_features"
HOLISTIC_MODEL = str(ROOT / "holistic_landmarker.task")

# ── Feature dimensions ────────────────────────────────────────────────────────
MAX_FRAMES   = 30
POSE_DIM     = 33 * 4    # 132

# 40 face key-points relevant to sign language (eyebrows, eyes, mouth).
# Nose-tip (landmark 1) is used as origin — not included in output.
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

SEQUENCES_PER_WORD = 20
PAUSE_EVERY        = 5    # pause after this many sequences so you can change lighting
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".MOV", ".MP4"}

# ── Target words (same 50 as video_processor.py) ─────────────────────────────
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
C_YELLOW = (0,   210, 255)
C_RED    = (50,  50,  220)
C_CYAN   = (255, 200, 0)
C_ORANGE = (0,   140, 255)


# ── Raw-video scanner (mirrors video_processor.py) ────────────────────────────

def _normalize(s: str) -> str:
    return re.sub(r"[\s_\-]", "", s).lower()


def _word_from_folder(name: str) -> str:
    m = re.match(r"^\d+\.\s*(.+)$", name)
    return m.group(1).strip() if m else name.strip()


def scan_raw_videos() -> dict[str, Path]:
    """Return {target_word: first_video_path} for every word we can match."""
    # Build normalised lookup: norm_key -> first video path inside that folder
    folder_map: dict[str, Path] = {}
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

    result: dict[str, Path] = {}
    for word in TARGET_WORDS:
        key = _normalize(word)
        if key in folder_map:
            result[word] = folder_map[key]
    return result


# ── MediaPipe ─────────────────────────────────────────────────────────────────

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


# ── Feature extraction with wrist-relative normalisation ──────────────────────

def _wrist_relative(lm_list) -> np.ndarray:
    coords = np.array([[lm.x, lm.y, lm.z] for lm in lm_list], dtype=np.float32)
    coords -= coords[0]          # wrist → (0,0,0); all fingers relative to wrist
    return coords.flatten()


def extract_keypoints(result: mv.HolisticLandmarkerResult) -> np.ndarray:
    if result.pose_landmarks:
        pose = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility] for lm in result.pose_landmarks],
            dtype=np.float32,
        ).flatten()
    else:
        pose = np.zeros(POSE_DIM, dtype=np.float32)

    if result.face_landmarks and len(result.face_landmarks) > max(FACE_KEY_INDICES):
        lms  = result.face_landmarks
        nose = np.array([lms[1].x, lms[1].y, lms[1].z], dtype=np.float32)
        pts  = np.array([[lms[i].x, lms[i].y, lms[i].z]
                          for i in FACE_KEY_INDICES], dtype=np.float32)
        pts -= nose   # nose-relative: expression shape, not position
        face = pts.flatten()
    else:
        face = np.zeros(FACE_DIM, dtype=np.float32)

    lh = _wrist_relative(result.left_hand_landmarks) \
        if result.left_hand_landmarks else np.zeros(HAND_DIM, dtype=np.float32)
    rh = _wrist_relative(result.right_hand_landmarks) \
        if result.right_hand_landmarks else np.zeros(HAND_DIM, dtype=np.float32)

    return np.concatenate([pose, face, lh, rh])


# ── Overlay helpers ───────────────────────────────────────────────────────────

def _banner(frame, text, y, colour, scale=0.9, thickness=2):
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0, y - 36), (w, y + 10), C_BLACK, -1)
    cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, colour, thickness)


def _progress_bar(frame, done, total):
    h, w = frame.shape[:2]
    bar_w = w - 20
    filled = int(bar_w * done / total)
    cv2.rectangle(frame, (10, h - 12), (10 + bar_w, h - 4), C_GRAY, 1)
    cv2.rectangle(frame, (10, h - 12), (10 + filled,  h - 4), C_GREEN, -1)


# ── Phase 1: tutorial (reference video loop) ──────────────────────────────────

def tutorial_phase(word: str, video_path: Path, word_idx: int) -> str:
    """
    Loop the reference video until the user presses C (continue), S (skip),
    or Q (quit).  Returns 'continue', 'skip', or 'quit'.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [!] Cannot open reference video: {video_path}")
        return "continue"       # no video — go straight to recording

    win = "Tutorial — Watch the sign"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 720, 540)

    action = "continue"
    while True:
        ok, frame = cap.read()
        if not ok:                       # end of video — loop back
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        h, w = frame.shape[:2]

        # word title
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (w, 54), C_BLACK, -1)
        cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)
        cv2.putText(frame, f"SIGN:  {word}",
                    (14, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.1, C_YELLOW, 2)

        # instruction
        _banner(frame, "Watch the gesture.   C = ready to record   S = skip   Q = quit",
                h - 14, C_WHITE, scale=0.55, thickness=1)

        # word progress
        cv2.putText(frame, f"Word {word_idx + 1} / {len(TARGET_WORDS)}",
                    (w - 160, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_GRAY, 1)

        cv2.imshow(win, frame)
        key = cv2.waitKey(30) & 0xFF   # ~33 fps playback

        if key == ord("c"):
            action = "continue"
            break
        elif key == ord("s"):
            action = "skip"
            break
        elif key == ord("q"):
            action = "quit"
            break

    cap.release()
    cv2.destroyWindow(win)
    return action


# ── Phase 2: webcam recording ─────────────────────────────────────────────────

def recording_phase(
    word: str,
    word_idx: int,
    holistic: mv.HolisticLandmarker,
    out_dir: Path,
    camera_index: int = 0,
) -> str:
    """
    Open webcam, record SEQUENCES_PER_WORD × 30-frame landmark sequences.
    Returns 'continue', 'skip', or 'quit'.
    """
    # find next available sequence index (allows re-running without overwriting)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("seq_*.npy"))
    next_idx  = len(existing)
    need      = SEQUENCES_PER_WORD - next_idx   # how many more to record this run

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_index}")

    win = "Personal Collector — Webcam"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 720, 540)

    seqs_recorded = 0
    state = "idle"    # idle | wait | lighting_break | recording | done
    current_frames: list[np.ndarray] = []
    wait_until = 0.0
    action = "continue"

    print(f"\n  [{word_idx + 1}/{len(TARGET_WORDS)}]  {word}")
    print(f"  Output → {out_dir}")
    print(f"  Already have {next_idx} sequences — recording {need} more "
          f"(target: {SEQUENCES_PER_WORD}) ...\n")

    while True:
        ok, raw = cap.read()
        if not ok:
            continue

        # mirror so right hand = right hand (matches training convention)
        frame = cv2.flip(raw, 1)
        h, w  = frame.shape[:2]

        # ── process landmarks only when actively recording ────────────
        if state in ("recording", "wait"):
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result   = holistic.detect(mp_image)
        else:
            result = None

        # ── state machine ─────────────────────────────────────────────
        if state == "recording":
            if result is not None:
                current_frames.append(extract_keypoints(result))

            if len(current_frames) == MAX_FRAMES:
                # save sequence
                seq_array = np.array(current_frames, dtype=np.float32)  # (30, 1692)
                save_path = out_dir / f"seq_{next_idx:04d}.npy"
                np.save(str(save_path), seq_array)
                seqs_recorded += 1
                next_idx      += 1
                current_frames = []
                print(f"    Saved seq {seqs_recorded}/{SEQUENCES_PER_WORD}  → {save_path.name}")

                if seqs_recorded >= need:
                    state = "done"
                elif seqs_recorded % PAUSE_EVERY == 0:
                    state = "lighting_break"   # pause to change lighting
                else:
                    state      = "wait"
                    wait_until = time.time() + 1.0

        elif state == "wait":
            if time.time() >= wait_until:
                state = "recording"

        # ── draw UI ───────────────────────────────────────────────────
        # top bar
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (w, 54), C_BLACK, -1)
        cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, f"SIGN:  {word}",
                    (14, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.1, C_YELLOW, 2)
        cv2.putText(frame, f"Word {word_idx + 1}/{len(TARGET_WORDS)}",
                    (w - 160, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_GRAY, 1)

        if state == "idle":
            _banner(frame,
                    f"Press R to record {need} more sequences ({next_idx}/{SEQUENCES_PER_WORD} done)   S = skip   Q = quit",
                    h - 14, C_WHITE, scale=0.55, thickness=1)

        elif state == "recording":
            prog = len(current_frames)
            # frame progress bar
            bar_w  = w - 20
            filled = int(bar_w * prog / MAX_FRAMES)
            cv2.rectangle(frame, (10, h - 20), (10 + bar_w, h - 8), C_GRAY, 1)
            cv2.rectangle(frame, (10, h - 20), (10 + filled, h - 8), C_GREEN, -1)

            label = (f"Recording {seqs_recorded + 1}/{need}"
                     f"  — frame {prog}/{MAX_FRAMES}")
            _banner(frame, label, h - 28, C_GREEN, scale=0.65)

            # big REC indicator
            cv2.circle(frame, (w - 30, 30), 10, C_RED, -1)
            cv2.putText(frame, "REC", (w - 60, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_RED, 2)

        elif state == "wait":
            remaining = max(0.0, wait_until - time.time())
            ov2 = frame.copy()
            cv2.rectangle(ov2, (0, h // 2 - 50), (w, h // 2 + 50), C_BLACK, -1)
            cv2.addWeighted(ov2, 0.7, frame, 0.3, 0, frame)
            cv2.putText(frame, f"WAIT — reset hands   ({remaining:.1f}s)",
                        (w // 2 - 210, h // 2 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, C_ORANGE, 2)
            _banner(frame,
                    f"Seq {seqs_recorded}/{need} done — get ready for next",
                    h - 14, C_GRAY, scale=0.55, thickness=1)

        elif state == "lighting_break":
            ov2 = frame.copy()
            cv2.rectangle(ov2, (0, h // 2 - 70), (w, h // 2 + 70), C_BLACK, -1)
            cv2.addWeighted(ov2, 0.75, frame, 0.25, 0, frame)
            cv2.putText(frame, f"{seqs_recorded}/{need} done",
                        (w // 2 - 100, h // 2 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, C_GREEN, 2)
            cv2.putText(frame, "Change lighting / position",
                        (w // 2 - 200, h // 2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, C_YELLOW, 2)
            cv2.putText(frame, "then press  R  to continue",
                        (w // 2 - 195, h // 2 + 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, C_WHITE, 1)

        elif state == "done":
            ov3 = frame.copy()
            cv2.rectangle(ov3, (0, h // 2 - 50), (w, h // 2 + 50), C_BLACK, -1)
            cv2.addWeighted(ov3, 0.7, frame, 0.3, 0, frame)
            cv2.putText(frame, f"'{word}'  COMPLETE!",
                        (w // 2 - 180, h // 2 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.95, C_GREEN, 2)
            _banner(frame, "SPACE = next word   Q = quit",
                    h - 14, C_WHITE, scale=0.55, thickness=1)
            # overall progress bar
            _progress_bar(frame, word_idx + 1, len(TARGET_WORDS))

        cv2.imshow(win, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            action = "quit"
            break
        elif key == ord("s") and state in ("idle",):
            action = "skip"
            break
        elif key == ord("r") and state in ("idle", "lighting_break"):
            state          = "recording"
            current_frames = []
        elif key == ord(" ") and state == "done":
            action = "continue"
            break

    cap.release()
    cv2.destroyWindow(win)
    return action


# ── Entry point ───────────────────────────────────────────────────────────────

def main(camera_index: int = 0) -> None:
    print("Scanning raw dataset for reference videos ...")
    video_map = scan_raw_videos()
    found = sum(1 for w in TARGET_WORDS if w in video_map)
    print(f"  Found reference videos for {found}/{len(TARGET_WORDS)} words.\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with build_holistic() as holistic:
        for idx, word in enumerate(TARGET_WORDS):
            word_out = OUT_DIR / word
            existing = list(word_out.glob("seq_*.npy")) if word_out.exists() else []

            if len(existing) >= SEQUENCES_PER_WORD:
                print(f"  [{idx+1}/{len(TARGET_WORDS)}] '{word}' — already has "
                      f"{len(existing)} sequences, skipping.")
                continue

            print(f"\n{'='*60}")
            print(f"  Word {idx+1}/{len(TARGET_WORDS)}: {word}")
            print(f"{'='*60}")

            # ── Tutorial ───────────────────────────────────────────────
            if word in video_map:
                print(f"  Reference video: {video_map[word].name}")
                action = tutorial_phase(word, video_map[word], idx)
            else:
                print(f"  No reference video found — skipping tutorial.")
                action = "continue"

            if action == "quit":
                print("\nQuitting.")
                break
            if action == "skip":
                print(f"  Skipped '{word}'.")
                continue

            # ── Recording ──────────────────────────────────────────────
            action = recording_phase(word, idx, holistic, word_out, camera_index)

            if action == "quit":
                print("\nQuitting.")
                break
            if action == "skip":
                print(f"  Skipped '{word}'.")
                continue

    # Summary
    print("\n" + "="*60)
    print("Collection summary:")
    total = 0
    for word in TARGET_WORDS:
        n = len(list((OUT_DIR / word).glob("seq_*.npy"))) if (OUT_DIR / word).exists() else 0
        if n:
            print(f"  {word:<25} {n} sequences")
            total += n
    print(f"\n  Total sequences saved: {total}")
    print(f"  Output directory:      {OUT_DIR}")


if __name__ == "__main__":
    main()
