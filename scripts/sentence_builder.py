"""
sentence_builder.py — Real-time ISL sentence builder with multilingual TTS.

Signs words one at a time; accumulates them into a sentence; injects
am / is / are automatically; translates to a selected Indian language
and speaks it aloud via gTTS + pygame.

Controls:
    1-6   — select output language (Hindi/Bengali/Tamil/Telugu/Marathi/Gujarati)
    ENTER — finalize sentence immediately and speak
    B     — undo last word
    C     — clear current sentence
    Q     — quit

Run:
    venv312\\Scripts\\python.exe sentence_builder.py
"""

import queue
import tempfile
import threading
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pygame
from deep_translator import GoogleTranslator
from groq import Groq
from gtts import gTTS
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision as mv
from tensorflow.keras.models import load_model

# ── Groq API ──────────────────────────────────────────────────────────────────
import os
try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL   = "llama-3.1-8b-instant"

_groq_client = Groq(api_key=GROQ_API_KEY)

# Multi-word ISL signs that must never be grammatically modified
FIXED_PHRASES = {"Good Morning", "How are you", "Thank you", "Hello"}


def enhance_grammar(words: list[str]) -> str:
    if not words:
        return "."
    # If the entire utterance is a fixed greeting phrase, return as-is — no LLM
    joined = " ".join(words)
    for phrase in FIXED_PHRASES:
        if joined.lower() == phrase.lower():
            return phrase + "."

    fallback = " ".join(inject_grammar(words)) + "."
    try:
        response = _groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You convert Indian Sign Language (ISL) gloss word sequences into natural English sentences. "
                        "STRICT RULES — follow exactly:\n"
                        "(1) Keep the original word order. Never reorder words.\n"
                        "(2) Only add linking words (am/is/are/a/an/in/the) if WITHOUT them the sentence makes no "
                        "grammatical sense. If the words already form a meaningful phrase, output them as-is.\n"
                        "(3) NEVER add linking words inside fixed greetings or multi-word phrases like "
                        "'Good Morning', 'How are you', 'Thank you', 'Hello'.\n"
                        "(4) Do NOT add any new nouns, verbs, or content words not present in the input.\n"
                        "(5) Reply with the final sentence only — no explanation, no quotes."
                    ),
                },
                {"role": "user",      "content": "Gloss words: I Teacher"},
                {"role": "assistant", "content": "I am a Teacher."},
                {"role": "user",      "content": "Gloss words: he Doctor"},
                {"role": "assistant", "content": "He is a Doctor."},
                {"role": "user",      "content": "Gloss words: Mother Hospital"},
                {"role": "assistant", "content": "Mother is in the Hospital."},
                {"role": "user",      "content": "Gloss words: we happy"},
                {"role": "assistant", "content": "We are happy."},
                {"role": "user",      "content": "Gloss words: Good Morning"},
                {"role": "assistant", "content": "Good Morning."},
                {"role": "user",      "content": "Gloss words: How are you"},
                {"role": "assistant", "content": "How are you."},
                {"role": "user",      "content": "Gloss words: Thank you"},
                {"role": "assistant", "content": "Thank you."},
                {"role": "user",      "content": f"Gloss words: {joined}"},
            ],
            temperature=0.1,
            max_tokens=60,
        )
        result = response.choices[0].message.content.strip()
        if not result.endswith("."):
            result += "."
        return result
    except Exception as e:
        print(f"  [Groq] {e} — using rule-based fallback")
        return fallback


# ── Language options ──────────────────────────────────────────────────────────
LANGUAGES = {
    "1": ("English",   "en"),
    "2": ("Hindi",     "hi"),
    "3": ("Bengali",   "bn"),
    "4": ("Tamil",     "ta"),
    "5": ("Telugu",    "te"),
    "6": ("Marathi",   "mr"),
    "7": ("Gujarati",  "gu"),
}
DEFAULT_LANG_KEY = "1"   # English on startup

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT           = Path(__file__).parent.parent   # project root (one level up from scripts/)
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
KEYPOINT_DIM = POSE_DIM + FACE_DIM + HAND_DIM * 2

# ── Sentence builder parameters ───────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.60   # min confidence to count a prediction
VOTE_WINDOW          = 6      # rolling window of recent predictions
VOTE_THRESHOLD       = 4      # 4/6 = 67% agreement to confirm a word
LOW_CONF_TOLERANCE   = 2      # consecutive low-conf frames before vote window resets
PREDICT_EVERY        = 2      # predict every 2nd frame — balanced CPU vs responsiveness
SENTENCE_TIMEOUT     = 2.5    # seconds of no hands → auto-finalize sentence
WORD_COOLDOWN        = 1.0    # seconds after confirming; buffer refreshes naturally here
DISPLAY_TIME         = 0.0    # unused — completed sentence stays until C is pressed

# ── Grammar rules ─────────────────────────────────────────────────────────────
# Pronouns and people words as subjects — determine linking verb
SUBJECTS_AM  = {"I"}
SUBJECTS_IS  = {"he", "she", "it", "Mother", "Father", "Brother", "Sister",
                "Friend", "Man", "Woman", "Teacher", "Doctor", "Police",
                "Student", "Patient"}
SUBJECTS_ARE = {"you", "we", "they", "Family"}
ALL_SUBJECTS = SUBJECTS_AM | SUBJECTS_IS | SUBJECTS_ARE

# Only pure pronouns block linking verb injection before the next word
# (professions like Teacher can follow a pronoun as a predicate noun)
PURE_PRONOUNS = {"I", "he", "she", "they", "we", "you", "it"}

# Predicate nouns that need "a" before them
ARTICLE_NOUNS = {"Teacher", "Doctor", "Police", "Student", "Patient",
                 "Man", "Woman", "Friend", "Dream", "Sign", "Sport"}

# Place nouns that need "in the" before them
PLACE_NOUNS = {"Hospital", "School", "House"}


def inject_grammar(words: list[str]) -> list[str]:
    """Insert am/is/are and articles/prepositions between subject and predicate."""
    if not words:
        return []
    result = []
    for i, word in enumerate(words):
        result.append(word)
        if i < len(words) - 1:
            nxt = words[i + 1]
            if nxt in PURE_PRONOUNS:
                continue
            if word in SUBJECTS_AM:
                result.append("am")
                if nxt in ARTICLE_NOUNS:   result.append("a")
                elif nxt in PLACE_NOUNS:   result.extend(["in", "the"])
            elif word in SUBJECTS_IS:
                result.append("is")
                if nxt in ARTICLE_NOUNS:   result.append("a")
                elif nxt in PLACE_NOUNS:   result.extend(["in", "the"])
            elif word in SUBJECTS_ARE:
                result.append("are")
                if nxt in ARTICLE_NOUNS:   result.append("a")
                elif nxt in PLACE_NOUNS:   result.extend(["in", "the"])
    return result


# ── Colours (BGR) ─────────────────────────────────────────────────────────────
C_BLACK  = (0,   0,   0)
C_WHITE  = (255, 255, 255)
C_GRAY   = (140, 140, 140)
C_GREEN  = (0,   210, 0)
C_RED    = (60,  60,  220)
C_YELLOW = (0,   215, 255)
C_CYAN   = (255, 200, 0)
C_ORANGE = (0,   140, 255)


# ── TTS engine ────────────────────────────────────────────────────────────────

pygame.mixer.init()
_tts_queue: queue.Queue = queue.Queue()


def _tts_worker() -> None:
    """Background thread: translates then speaks every sentence pushed to the queue."""
    while True:
        item = _tts_queue.get()
        if item is None:          # shutdown signal
            break
        english_text, lang_code = item
        try:
            translated = GoogleTranslator(source="en", target=lang_code).translate(english_text)
            tts = gTTS(text=translated, lang=lang_code, slow=False)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tmp_path = f.name
            tts.save(tmp_path)
            pygame.mixer.music.load(tmp_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.05)
        except Exception as e:
            print(f"  [TTS] {e}")
        finally:
            _tts_queue.task_done()


def speak(english_text: str, lang_code: str) -> None:
    """Non-blocking: push sentence to TTS queue."""
    _tts_queue.put((english_text, lang_code))


# ── MediaPipe setup ───────────────────────────────────────────────────────────

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

def _wrist_relative(lm_list) -> np.ndarray:
    coords = np.array([[lm.x, lm.y, lm.z] for lm in lm_list], dtype=np.float32)
    coords -= coords[0]
    return coords.flatten()

def extract_keypoints(result) -> np.ndarray:
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


# ── UI helpers ────────────────────────────────────────────────────────────────

def _dark_bar(frame, y0, y1, alpha=0.65):
    ov = frame.copy()
    cv2.rectangle(ov, (0, y0), (frame.shape[1], y1), C_BLACK, -1)
    cv2.addWeighted(ov, alpha, frame, 1 - alpha, 0, frame)

def _wrap_text(text, font, scale, thickness, max_w):
    """Split text into lines that fit within max_w pixels."""
    words = text.split()
    lines, current = [], ""
    for w in words:
        test = (current + " " + w).strip()
        (tw, _), _ = cv2.getTextSize(test, font, scale, thickness)
        if tw <= max_w:
            current = test
        else:
            if current:
                lines.append(current)
            current = w
    if current:
        lines.append(current)
    return lines


# ── Main ──────────────────────────────────────────────────────────────────────

def run(camera_index: int = 0) -> None:
    # ── Start TTS background thread ───────────────────────────────────────
    tts_thread = threading.Thread(target=_tts_worker, daemon=True)
    tts_thread.start()

    print("Loading labels ...")
    labels = sorted(p.name for p in FEATURES_DIR.iterdir() if p.is_dir())
    print(f"  {len(labels)} classes.\n")

    print("Loading model ...")
    model = load_model(KERAS_MODEL)
    print("  Done.\n")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_index}")

    cv2.namedWindow("ISL Sentence Builder", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ISL Sentence Builder", 900, 640)

    sequence: deque[np.ndarray] = deque(maxlen=MAX_FRAMES)

    sentence_words: list[str] = []
    vote_window        = deque(maxlen=VOTE_WINDOW)
    cooldown_until     = 0.0
    last_hand_time     = 0.0
    completed_sentence = ""
    completed_until    = 0.0
    frame_count        = 0
    current_prediction = ""
    current_conf       = 0.0
    top_vote_word      = ""
    top_vote_count     = 0
    low_conf_streak    = 0
    lang_key           = DEFAULT_LANG_KEY   # current output language

    with build_holistic() as holistic:
        while True:
            ok, raw = cap.read()
            if not ok:
                continue

            frame = cv2.flip(raw, 1)
            h, w  = frame.shape[:2]
            now   = time.time()

            # ── Landmark extraction ───────────────────────────────────
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = holistic.detect(mp_img)

            hands_present = bool(
                result.left_hand_landmarks or result.right_hand_landmarks
            )

            # ── Buffer management ─────────────────────────────────────
            if hands_present:
                last_hand_time = now
                sequence.append(extract_keypoints(result))
            else:
                # Clear vote window immediately — no hands means no valid sign
                vote_window.clear()
                low_conf_streak = 0
                if now - last_hand_time > 0.5:
                    sequence.clear()

            # ── Finalize sentence (shared helper) ────────────────────
            def _finalize():
                nonlocal completed_sentence, completed_until, sentence_words
                completed_sentence = enhance_grammar(sentence_words)
                completed_until    = float("inf")   # stays until C is pressed
                lang_name, lang_code = LANGUAGES[lang_key]
                print(f"\nSentence: {completed_sentence}  [{lang_name}]")
                speak(completed_sentence, lang_code)
                sentence_words = []
                sequence.clear()
                vote_window.clear()

            # ── Finalize sentence on timeout ──────────────────────────
            if (sentence_words and
                    now - last_hand_time > SENTENCE_TIMEOUT and
                    now > cooldown_until):
                _finalize()

            # ── Prediction & word confirmation ────────────────────────
            frame_count += 1

            if (len(sequence) == MAX_FRAMES and
                    now > cooldown_until and
                    frame_count % PREDICT_EVERY == 0 and
                    hands_present):

                X     = np.expand_dims(np.array(sequence, dtype=np.float32), 0)
                probs = model.predict(X, verbose=0)[0]
                top_i = int(np.argmax(probs))
                current_conf       = float(probs[top_i])
                current_prediction = labels[top_i]

                if current_conf >= CONFIDENCE_THRESHOLD:
                    vote_window.append(current_prediction)
                    low_conf_streak = 0
                else:
                    low_conf_streak += 1
                    if low_conf_streak >= LOW_CONF_TOLERANCE:
                        vote_window.clear()
                        low_conf_streak = 0

                if vote_window:
                    top_vote_word, top_vote_count = Counter(vote_window).most_common(1)[0]
                else:
                    top_vote_word, top_vote_count = "", 0

                if top_vote_count >= VOTE_THRESHOLD:
                    if not sentence_words or sentence_words[-1] != top_vote_word:
                        sentence_words.append(top_vote_word)
                        print(f"  + {top_vote_word}")
                    cooldown_until = now + WORD_COOLDOWN
                    vote_window.clear()
                    low_conf_streak = 0
                    # No sequence.clear() — deque naturally rolls over during cooldown.
                    # At 30 fps, WORD_COOLDOWN=1.0s pushes out all previous frames
                    # so the buffer is fresh the moment cooldown ends.

            # ── Draw UI ───────────────────────────────────────────────
            bar_w = w - 20

            # ── top bar: word + confidence (compact, 44px tall) ───────
            _dark_bar(frame, 0, 44)
            buf_ready = len(sequence) == MAX_FRAMES
            if not hands_present:
                cv2.putText(frame, "Waiting ...",
                            (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, C_GRAY, 2)
            elif not buf_ready:
                # buffer refilling — show last known prediction very dimly as context
                if current_prediction:
                    cv2.putText(frame, current_prediction,
                                (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (70, 70, 70), 2)
                buf_label = f"{len(sequence)}/{MAX_FRAMES}"
                cv2.putText(frame, buf_label,
                            (w - 90, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (70, 70, 70), 1)
            else:
                # buffer full — show live prediction with confidence
                colour = C_GREEN if current_conf >= CONFIDENCE_THRESHOLD else C_GRAY
                cv2.putText(frame, current_prediction,
                            (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 2)
                cv2.putText(frame, f"{current_conf*100:.0f}%",
                            (w - 80, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)

            # single vote progress bar — vote progress when active, buffer fill otherwise
            in_cooldown = now < cooldown_until
            if top_vote_count > 0 and not in_cooldown:
                vote_pct   = top_vote_count / VOTE_WINDOW
                bar_colour = C_ORANGE
            else:
                vote_pct   = len(sequence) / MAX_FRAMES
                bar_colour = C_CYAN
            cv2.rectangle(frame, (10, 46), (10 + bar_w, 50), C_GRAY, 1)
            cv2.rectangle(frame, (10, 46),
                          (10 + int(bar_w * min(vote_pct, 1.0)), 50),
                          bar_colour, -1)

            # ── bottom strip (70px: sentence + lang indicator + controls) ──
            _dark_bar(frame, h - 70, h, 0.75)

            lang_name, _ = LANGUAGES[lang_key]
            lang_label   = f"Lang: {lang_name}  [1=En 2=Hi 3=Bn 4=Ta 5=Te 6=Mr 7=Gu]"

            building = " ".join(sentence_words)
            if now < completed_until and completed_sentence:
                cv2.putText(frame, completed_sentence,
                            (12, h - 46), cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_GREEN, 2)
                cv2.putText(frame, lang_label,
                            (12, h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_CYAN, 1)
                cv2.putText(frame, "1-7 = change language  |  ENTER = speak  |  C = clear  |  Q = quit",
                            (12, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_GRAY, 1)
            elif building:
                cv2.putText(frame, building,
                            (12, h - 46), cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_YELLOW, 2)
                cv2.putText(frame, lang_label,
                            (12, h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_CYAN, 1)
                if not hands_present:
                    remaining = max(0.0, SENTENCE_TIMEOUT - (now - last_hand_time))
                    cv2.putText(frame,
                                f"Finalizing in {remaining:.1f}s  |  ENTER=speak now  |  B=undo  |  C=clear",
                                (12, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_ORANGE, 1)
                else:
                    cv2.putText(frame, "ENTER = speak now  |  B = undo  |  C = clear  |  Q = quit",
                                (12, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_GRAY, 1)
            else:
                cv2.putText(frame, "Start signing ...",
                            (12, h - 46), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_GRAY, 1)
                cv2.putText(frame, lang_label,
                            (12, h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_CYAN, 1)
                cv2.putText(frame, "ENTER = speak  |  C = clear  |  Q = quit",
                            (12, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_GRAY, 1)

            cv2.imshow("ISL Sentence Builder", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == 13:   # Enter — finalize if signing, or replay completed sentence
                if sentence_words:
                    _finalize()
                elif completed_sentence:
                    lang_name, lang_code = LANGUAGES[lang_key]
                    print(f"  Replaying in {lang_name}: {completed_sentence}")
                    speak(completed_sentence, lang_code)
            elif key == ord("b") and sentence_words:
                removed = sentence_words.pop()
                print(f"  - {removed} (undone)")
                vote_window.clear()
            elif key == ord("c"):
                sentence_words     = []
                vote_window.clear()
                completed_sentence = ""
                sequence.clear()
            elif key < 128 and chr(key) in LANGUAGES:
                lang_key = chr(key)
                print(f"  Language → {LANGUAGES[lang_key][0]}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
