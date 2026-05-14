"""
app.py — Flask backend for ISL Translator.

Runs the camera server-side (same pipeline as sentence_builder.py),
streams annotated video as MJPEG to the browser, and exposes
/status + /control endpoints for the frontend UI.

Run:
    venv312\\Scripts\\python.exe app.py
Then open: http://localhost:5000
"""

import os
import queue
# Load .env if present
try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass
import tempfile
import threading
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision as mv
from tensorflow.keras.models import load_model

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT           = Path(__file__).parent
HOLISTIC_MODEL = str(ROOT / "mediapipe_models" / "holistic_landmarker.task")
KERAS_MODEL    = str(ROOT / "models" / "isl_model_solo.keras")
FEATURES_DIR   = ROOT / "data" / "extracted_features"
FRONTEND_DIR   = ROOT / "Frontend"

# ── Feature constants ─────────────────────────────────────────────────────────
MAX_FRAMES = 30
POSE_DIM   = 33 * 4
FACE_KEY_INDICES = [
    70, 63, 105, 66, 107, 55, 65, 52,
    300, 293, 334, 296, 336, 285, 295, 282,
    33, 133, 159, 145, 362, 263, 386, 374,
    61, 37, 0, 267, 291, 321, 314, 17,
    78, 81, 13, 311, 308, 317, 14, 87,
]
FACE_DIM     = len(FACE_KEY_INDICES) * 3
HAND_DIM     = 21 * 3
KEYPOINT_DIM = POSE_DIM + FACE_DIM + HAND_DIM * 2

# ── Sentence builder params ───────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.60
VOTE_WINDOW          = 6
VOTE_THRESHOLD       = 4
LOW_CONF_TOLERANCE   = 2
PREDICT_EVERY        = 2
SENTENCE_TIMEOUT     = 2.5
WORD_COOLDOWN        = 1.0

# ── Languages ─────────────────────────────────────────────────────────────────
LANGUAGES = {
    "en": "English", "hi": "Hindi",  "bn": "Bengali",
    "ta": "Tamil",   "te": "Telugu", "mr": "Marathi", "gu": "Gujarati",
}

# ── Overlay colours (BGR) ─────────────────────────────────────────────────────
C_GREEN  = (80, 220, 120)
C_ORANGE = (60, 160, 255)
C_CYAN   = (220, 200, 80)
C_YELLOW = (60, 220, 240)
C_GRAY   = (160, 160, 160)
C_WHITE  = (240, 240, 240)


def _dark_bar(frame, y1, y2, alpha=0.65):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y1), (w, y2), (20, 20, 20), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


# ── Grammar helpers ───────────────────────────────────────────────────────────
# Multi-word ISL signs that must never be grammatically modified
FIXED_PHRASES = {"Good Morning", "How are you", "Thank you"}

SUBJECTS_AM  = {"I"}
SUBJECTS_IS  = {"he", "she", "it", "Mother", "Father", "Brother", "Sister",
                "Friend", "Man", "Woman", "Teacher", "Doctor", "Police",
                "Student", "Patient"}
SUBJECTS_ARE = {"you", "we", "they", "Family"}
PURE_PRONOUNS  = {"I", "he", "she", "they", "we", "you", "it"}
ARTICLE_NOUNS  = {"Teacher", "Doctor", "Police", "Student", "Patient",
                  "Man", "Woman", "Friend", "Dream", "Sign", "Sport"}
PLACE_NOUNS    = {"Hospital", "School", "House"}


def inject_grammar(words):
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
                if nxt in ARTICLE_NOUNS:  result.append("a")
                elif nxt in PLACE_NOUNS:  result.extend(["in", "the"])
            elif word in SUBJECTS_IS:
                result.append("is")
                if nxt in ARTICLE_NOUNS:  result.append("a")
                elif nxt in PLACE_NOUNS:  result.extend(["in", "the"])
            elif word in SUBJECTS_ARE:
                result.append("are")
                if nxt in ARTICLE_NOUNS:  result.append("a")
                elif nxt in PLACE_NOUNS:  result.extend(["in", "the"])
    return result


def enhance_grammar(words):
    if not words:
        return "."
    # Single fixed phrase — return exactly as signed, no LLM needed
    # Check both single-item and joined form, case-insensitive
    joined = " ".join(words)
    print(f"  [Grammar] words={words!r}  joined={joined!r}")
    for phrase in FIXED_PHRASES:
        if joined.lower() == phrase.lower() or (len(words) == 1 and words[0].lower() == phrase.lower()):
            return phrase + "."
    # Multi-word sentence containing a fixed phrase — protect it from rewriting
    # by passing it through but adding a strong hint to the LLM
    fallback = " ".join(inject_grammar(words)) + "."
    try:
        from groq import Groq
        client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": (
                    "You convert Indian Sign Language (ISL) gloss word sequences into natural English sentences. "
                    "STRICT RULES — follow exactly:\n"
                    "(1) Keep the original word order. Never reorder words.\n"
                    "(2) Only add linking words (am/is/are/a/an/in/the) if WITHOUT them the sentence makes no grammatical sense. "
                    "If the words already form a meaningful phrase, output them as-is.\n"
                    "(3) NEVER add linking words inside fixed greetings or multi-word phrases like "
                    "'Good Morning', 'How are you', 'Thank you', 'Good Night'.\n"
                    "(4) Do NOT add any new nouns, verbs, or content words not present in the input.\n"
                    "(5) Reply with the final sentence only — no explanation, no quotes, no commentary."
                )},
                {"role": "user",      "content": "Gloss words: I Teacher"},
                {"role": "assistant", "content": "I am a Teacher."},
                {"role": "user",      "content": "Gloss words: he Doctor"},
                {"role": "assistant", "content": "He is a Doctor."},
                {"role": "user",      "content": "Gloss words: Mother Hospital"},
                {"role": "assistant", "content": "Mother is in the Hospital."},
                {"role": "user",      "content": "Gloss words: Good Morning"},
                {"role": "assistant", "content": "Good Morning."},
                {"role": "user",      "content": "Gloss words: How are you"},
                {"role": "assistant", "content": "How are you."},
                {"role": "user",      "content": "Gloss words: Thank you"},
                {"role": "assistant", "content": "Thank you."},
                {"role": "user",      "content": f"Gloss words: {' '.join(words)}"},
            ],
            temperature=0.1, max_tokens=60,
        )
        result = response.choices[0].message.content.strip()
        if not result.endswith("."): result += "."
        return result
    except Exception as e:
        print(f"  [Groq] {e} — using rule-based fallback")
        return fallback


# ── TTS ───────────────────────────────────────────────────────────────────────
_tts_queue: queue.Queue = queue.Queue()


def _tts_worker():
    import pygame
    from deep_translator import GoogleTranslator
    from gtts import gTTS
    pygame.mixer.init()
    while True:
        item = _tts_queue.get()
        if item is None:
            break
        english_text, lang_code = item
        try:
            translated = english_text if lang_code == "en" else \
                GoogleTranslator(source="en", target=lang_code).translate(english_text)
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


def speak(text, lang_code):
    _tts_queue.put((text, lang_code))


# ── MediaPipe helpers ─────────────────────────────────────────────────────────
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


def _build_holistic():
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


# ── Global state ──────────────────────────────────────────────────────────────
_state = {
    "sequence":           deque(maxlen=MAX_FRAMES),
    "sentence_words":     [],
    "vote_window":        deque(maxlen=VOTE_WINDOW),
    "cooldown_until":     0.0,
    "last_hand_time":     0.0,
    "completed_sentence": "",
    "frame_count":        0,
    "current_word":       "",
    "current_conf":       0.0,
    "top_vote_word":      "",
    "top_vote_count":     0,
    "low_conf_streak":    0,
    "lang_code":          "en",
    "camera_active":      False,
    "hands_present":      False,
}

_labels          = []
_model           = None
_holistic        = None   # loaded once at startup, reused by camera thread
_current_frame   = None   # latest annotated JPEG bytes for MJPEG stream
_lock            = threading.Lock()
_camera_thread   = None

# Dark placeholder frame sent while camera is not yet started
def _make_placeholder():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (18, 26, 42)   # dark navy
    cv2.putText(img, "Camera not started", (160, 230),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (74, 120, 155), 2)
    cv2.putText(img, "Click 'Start Camera'", (170, 270),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 90, 120), 1)
    _, jpg = cv2.imencode(".jpg", img)
    return jpg.tobytes()

_BLACK_FRAME = _make_placeholder()


def _load_resources():
    global _labels, _model
    print("Loading labels ...")
    _labels = sorted(p.name for p in FEATURES_DIR.iterdir() if p.is_dir())
    print(f"  {len(_labels)} classes.")
    print("Loading Keras model ...")
    _model = load_model(KERAS_MODEL)
    print("  Done.\n")


def _do_finalize(s, now):
    s["completed_sentence"] = enhance_grammar(s["sentence_words"])
    lang_code = s["lang_code"]
    print(f"\nSentence: {s['completed_sentence']}  [{LANGUAGES.get(lang_code)}]")
    speak(s["completed_sentence"], lang_code)
    s["sentence_words"] = []
    s["sequence"].clear()
    s["vote_window"].clear()


# ── Camera loop (runs in background thread) ───────────────────────────────────
def _camera_loop(camera_index=0):
    global _current_frame

    print("  Opening camera ...")
    # Try DirectShow first (most reliable on Windows), fall back to default
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("  DirectShow failed, trying default backend ...")
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("  [ERROR] Cannot open camera at index", camera_index)
        with _lock:
            _state["camera_active"] = False
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Read one test frame to confirm camera works
    ok, test = cap.read()
    if not ok or test is None:
        print("  [ERROR] Camera opened but cannot read frames.")
        cap.release()
        with _lock:
            _state["camera_active"] = False
        return

    print("  Camera opened OK — frame size:", test.shape)

    # Build holistic fresh inside this thread — MediaPipe is not cross-thread safe
    print("  Loading MediaPipe in camera thread ...")
    holistic = _build_holistic()
    print("  Camera running.\n")

    sequence      = deque(maxlen=MAX_FRAMES)
    vote_window   = deque(maxlen=VOTE_WINDOW)
    last_hand_t   = 0.0
    cooldown_t    = 0.0
    frame_count   = 0
    cur_word      = ""
    cur_conf      = 0.0
    top_vote_word = ""
    top_vote_cnt  = 0
    low_conf_s    = 0

    while True:
        with _lock:
            if not _state["camera_active"]:
                break
            # pull mutable state we need
            sentence_words     = _state["sentence_words"]
            completed_sentence = _state["completed_sentence"]
            lang_code          = _state["lang_code"]

        ok, raw = cap.read()
        if not ok:
            continue

        frame = cv2.flip(raw, 1)
        h, w  = frame.shape[:2]
        now   = time.time()

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = holistic.detect(mp_img)

        hands_present = bool(result.left_hand_landmarks or result.right_hand_landmarks)

        if hands_present:
            last_hand_t = now
            sequence.append(extract_keypoints(result))
        else:
            if now - last_hand_t > 0.5:
                sequence.clear()
                vote_window.clear()

        # Auto-finalize
        if sentence_words and now - last_hand_t > SENTENCE_TIMEOUT and now > cooldown_t:
            with _lock:
                _do_finalize(_state, now)
                sentence_words = _state["sentence_words"]
                completed_sentence = _state["completed_sentence"]
            sequence.clear()
            vote_window.clear()

        # Predict
        frame_count += 1
        if len(sequence) == MAX_FRAMES and now > cooldown_t and frame_count % PREDICT_EVERY == 0:
            X     = np.expand_dims(np.array(sequence, dtype=np.float32), 0)
            probs = _model.predict(X, verbose=0)[0]
            top_i = int(np.argmax(probs))
            cur_conf = float(probs[top_i])
            cur_word = _labels[top_i]

            if cur_conf >= CONFIDENCE_THRESHOLD:
                vote_window.append(cur_word)
                low_conf_s = 0
            else:
                low_conf_s += 1
                if low_conf_s >= LOW_CONF_TOLERANCE:
                    vote_window.clear()
                    low_conf_s = 0

            if vote_window:
                top_vote_word, top_vote_cnt = Counter(vote_window).most_common(1)[0]
            else:
                top_vote_word, top_vote_cnt = "", 0

            if top_vote_cnt >= VOTE_THRESHOLD:
                with _lock:
                    if not _state["sentence_words"] or _state["sentence_words"][-1] != top_vote_word:
                        _state["sentence_words"].append(top_vote_word)
                        sentence_words = _state["sentence_words"]
                        print(f"  + {top_vote_word}")
                cooldown_t = now + WORD_COOLDOWN
                vote_window.clear()
                low_conf_s = 0

        # ── Update shared state ───────────────────────────────────────────────
        buf_pct  = len(sequence) / MAX_FRAMES
        vote_pct = top_vote_cnt / VOTE_WINDOW if top_vote_cnt > 0 else 0
        with _lock:
            _state["current_word"]    = cur_word
            _state["current_conf"]    = cur_conf
            _state["top_vote_word"]   = top_vote_word
            _state["top_vote_count"]  = top_vote_cnt
            _state["hands_present"]   = hands_present
            _state["buf_pct"]         = round(buf_pct, 2)
            _state["vote_pct"]        = round(vote_pct, 2)
            _state["in_cooldown"]     = now < cooldown_t

        # ── Draw overlays (same as sentence_builder.py) ───────────────────────
        bar_w = w - 20

        # top bar
        _dark_bar(frame, 0, 48)
        buf_ready = len(sequence) == MAX_FRAMES
        if not hands_present:
            cv2.putText(frame, "Waiting ...", (12, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, C_GRAY, 2)
        elif not buf_ready:
            if cur_word:
                cv2.putText(frame, cur_word, (12, 34),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (70, 70, 70), 2)
            cv2.putText(frame, f"{len(sequence)}/{MAX_FRAMES}",
                        (w - 90, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (70, 70, 70), 1)
        else:
            colour = C_GREEN if cur_conf >= CONFIDENCE_THRESHOLD else C_GRAY
            cv2.putText(frame, cur_word, (12, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 2)
            cv2.putText(frame, f"{cur_conf*100:.0f}%",
                        (w - 80, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)

        # progress bar
        in_cooldown = now < cooldown_t
        if top_vote_cnt > 0 and not in_cooldown:
            bar_col = C_ORANGE
            bar_pct = top_vote_cnt / VOTE_WINDOW
        else:
            bar_col = C_CYAN
            bar_pct = len(sequence) / MAX_FRAMES
        cv2.rectangle(frame, (10, 50), (10 + bar_w, 54), C_GRAY, 1)
        cv2.rectangle(frame, (10, 50),
                      (10 + int(bar_w * min(bar_pct, 1.0)), 54), bar_col, -1)

        # bottom strip
        _dark_bar(frame, h - 72, h, 0.75)
        with _lock:
            sw = _state["sentence_words"][:]
            cs = _state["completed_sentence"]
        lang_label = f"Lang: {LANGUAGES.get(lang_code, 'English')}  [1=En 2=Hi 3=Bn 4=Ta 5=Te 6=Mr 7=Gu]"
        building = " ".join(sw)

        if cs and not sw:
            cv2.putText(frame, cs, (12, h - 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_GREEN, 2)
        elif building:
            cv2.putText(frame, building, (12, h - 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_YELLOW, 2)
        else:
            cv2.putText(frame, "Start signing ...", (12, h - 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_GRAY, 1)

        cv2.putText(frame, lang_label, (12, h - 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_CYAN, 1)
        cv2.putText(frame, "Use website controls to Confirm / Speak / Undo / Clear",
                    (12, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_GRAY, 1)

        # Encode to JPEG and store
        _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        _current_frame = jpg.tobytes()

    cap.release()
    # Do NOT close holistic — it is the global singleton, reused on next start
    _current_frame = None
    print("  Camera stopped.")


# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
CORS(app)


@app.route("/")
def index():
    return send_from_directory(str(FRONTEND_DIR), "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(str(FRONTEND_DIR), filename)


@app.route("/video_feed")
def video_feed():
    """MJPEG stream — point an <img> tag at this endpoint."""
    def generate():
        while True:
            frame = _current_frame if _current_frame is not None else _BLACK_FRAME
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(0.033)   # ~30 fps
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    """Current recognition state — polled by the frontend."""
    with _lock:
        s = _state
        return jsonify({
            "camera_active":      s["camera_active"],
            "hands":              s.get("hands_present", False),
            "current_word":       s["current_word"],
            "current_conf":       round(s["current_conf"] * 100),
            "top_vote_word":      s["top_vote_word"],
            "top_vote_count":     s["top_vote_count"],
            "sentence_words":     s["sentence_words"][:],
            "completed_sentence": s["completed_sentence"],
            "buf_pct":            s.get("buf_pct", 0),
            "vote_pct":           s.get("vote_pct", 0),
            "in_cooldown":        s.get("in_cooldown", False),
            "lang_code":          s["lang_code"],
            "lang_name":          LANGUAGES.get(s["lang_code"], "English"),
        })


@app.route("/control", methods=["POST"])
def control():
    global _camera_thread
    data   = request.get_json(silent=True) or {}
    action = data.get("action", "")

    with _lock:
        s   = _state
        now = time.time()

        if action == "start_camera":
            if not s["camera_active"]:
                s["camera_active"] = True
                t = threading.Thread(target=_camera_loop, daemon=True)
                t.start()
                _camera_thread = t

        elif action == "stop_camera":
            s["camera_active"] = False
            s["sequence"].clear()
            s["vote_window"].clear()
            s["current_word"]  = ""
            s["current_conf"]  = 0.0

        elif action == "finalize":
            if s["sentence_words"]:
                _do_finalize(s, now)
            elif s["completed_sentence"]:
                speak(s["completed_sentence"], s["lang_code"])

        elif action == "clear":
            s["sentence_words"]     = []
            s["vote_window"].clear()
            s["completed_sentence"] = ""
            s["sequence"].clear()
            s["current_word"]       = ""
            s["current_conf"]       = 0.0

        elif action == "undo":
            if s["sentence_words"]:
                removed = s["sentence_words"].pop()
                print(f"  - {removed} (undone)")
                s["vote_window"].clear()

        elif action == "language":
            lang = data.get("lang", "en")
            if lang in LANGUAGES:
                s["lang_code"] = lang
                print(f"  Language: {LANGUAGES[lang]}")

        elif action == "speak":
            if s["completed_sentence"]:
                speak(s["completed_sentence"], s["lang_code"])

        return jsonify({
            "camera_active":      s["camera_active"],
            "sentence_words":     s["sentence_words"][:],
            "completed_sentence": s["completed_sentence"],
            "lang_code":          s["lang_code"],
            "lang_name":          LANGUAGES.get(s["lang_code"], "English"),
        })


if __name__ == "__main__":
    _load_resources()

    tts_thread = threading.Thread(target=_tts_worker, daemon=True)
    tts_thread.start()

    print("Starting Flask server at http://localhost:5000")
    print("Open http://localhost:5000 in your browser.\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
