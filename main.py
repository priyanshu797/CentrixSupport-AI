from __future__ import annotations

import json
import logging
import os
import re
import smtplib
import time
import traceback
from collections.abc import Generator
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import cast

from flask import (
    Flask,
    Response,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    stream_with_context,
    url_for,
)
from flask_cors import CORS
from werkzeug.utils import secure_filename

import prompt

# ── Module logger (fixes LOG015) ───────────────────────────────────────────────
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# ── Optional heavy dependencies ────────────────────────────────────────────────
try:
    import nltk

    def _safe_nltk_download() -> None:
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)

    _safe_nltk_download()
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    import importlib.util

    RAG_AVAILABLE = (
        importlib.util.find_spec("llama_index") is not None
        and importlib.util.find_spec("llama_index.embeddings.huggingface") is not None
    )
    if RAG_AVAILABLE:
        logger.info("LlamaIndex + HuggingFace embedding available")
except Exception as exc:  # noqa: BLE001
    RAG_AVAILABLE = False
    logger.warning("RAG availability check failed: %s", exc)

try:
    from groq import Groq
    from groq.types.chat import ChatCompletionMessageParam

    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logger.warning("groq SDK not installed")

# ── Agentic RAG imports ────────────────────────────────────────────────────────
try:
    from app.content_retrieval import (
        DOC_RELEVANCE_THRESHOLD,  # noqa: F401 — re-exported for tests
        Agent,
        MultiLevelCache,  # noqa: F401
        QueryClassifier,  # noqa: F401
        Tools,  # noqa: F401
    )
    from app.indexing_service import DocumentIndexingService
    from app.content_retrieval import (
        GROQ_API_KEY as RAG_GROQ_API_KEY,  # noqa: F401
    )
    from app.content_retrieval import (
        format_response as rag_format_response,  # noqa: F401
    )
    from app.content_retrieval import (
        initialize_settings as rag_initialize_settings,
    )

    AGENTIC_RAG_AVAILABLE = True
    logger.info("Agentic RAG module loaded")
except Exception as exc:  # noqa: BLE001 — intentional broad catch at import time
    AGENTIC_RAG_AVAILABLE = False
    logger.warning("Agentic RAG module unavailable: %s", exc)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
CONVERSATION_FOLDER = os.path.join(BASE_DIR, "conversations")
RAG_CACHE_DIR = os.path.join(BASE_DIR, "rag_cache")

ALLOWED_EXTENSIONS = {
    "pdf",
    "txt",
    "docx",
    "doc",
    "png",
    "jpg",
    "jpeg",
    "bmp",
    "gif",
    "tiff",
    "csv",
    "json",
    "xml",
    "html",
    "htm",
    "md",
    "mp3",
    "wav",
    "mp4",
}

# ── API key ────────────────────────────────────────────────────────────────────
GROQ_API_KEY: str = "enter_api_key"

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    static_url_path="/static",
    static_folder=os.path.join(BASE_DIR, "static"),
    template_folder=os.path.join(BASE_DIR, "templates"),
)
CORS(app)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev_secret_key_change_in_production")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["CONVERSATION_FOLDER"] = CONVERSATION_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 150 * 1024 * 1024

for _d in [UPLOAD_FOLDER, CONVERSATION_FOLDER, RAG_CACHE_DIR]:
    os.makedirs(_d, exist_ok=True)

# ── Groq client ────────────────────────────────────────────────────────────────
groq_client = None
if GROQ_AVAILABLE and GROQ_API_KEY:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        logger.info("Groq client initialised")
    except Exception as exc:  # noqa: BLE001
        logger.error("Groq init failed: %s", exc)

# Email config
SENDER_EMAIL: str | None = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD: str | None = os.getenv("SENDER_PASSWORD")
RECEIVER_EMAIL: str | None = os.getenv("RECEIVER_EMAIL", SENDER_EMAIL)

# ── Initialise RAG settings once at startup ────────────────────────────────────
_rag_settings_initialised = False
if AGENTIC_RAG_AVAILABLE and RAG_AVAILABLE:
    try:
        rag_initialize_settings()
        _rag_settings_initialised = True
        logger.info("Agentic RAG settings initialised")
    except Exception as exc:  # noqa: BLE001
        logger.error("RAG settings init failed: %s", exc)


_indexing_service: DocumentIndexingService | None = None
if _rag_settings_initialised:
    try:
        _indexing_service = DocumentIndexingService(
            os.path.join(BASE_DIR, "rag_storage")
        )
        logger.info("Asynchronous document indexing service ready")
    except Exception as exc:  # noqa: BLE001 — optional service startup
        logger.error("Document indexing service init failed: %s", exc)


# ── Language / emotion helpers ─────────────────────────────────────────────────
def detect_language(text: str) -> str:
    cleaned = re.sub(r"[^\w\s]", "", text.lower())
    hindi_pattern = re.compile(r"[\u0900-\u097F]")
    hinglish_words = {
        "kya",
        "hai",
        "hoon",
        "hain",
        "aur",
        "ki",
        "ka",
        "ke",
        "ko",
        "mein",
        "main",
        "se",
        "par",
        "theek",
        "nahi",
        "haan",
        "bahut",
        "bohot",
    }
    english_words = {
        "the",
        "is",
        "are",
        "was",
        "were",
        "what",
        "when",
        "where",
        "how",
        "feel",
        "feeling",
        "help",
        "need",
    }
    hindi_chars = len(hindi_pattern.findall(text))
    total_chars = len(re.sub(r"\s", "", text))
    if total_chars and (hindi_chars / total_chars) > 0.3:
        return "hindi"
    words = set(cleaned.split())
    h = len(words & hinglish_words)
    e = len(words & english_words)
    if h and e:
        return "hinglish"
    if h > e:
        return "hinglish"
    if hindi_chars:
        return "hindi"
    return "english"


def detect_emotion(text: str) -> tuple[str, float]:
    tl = text.lower()
    groups: dict[str, list[str]] = {
        "overwhelmed": ["overwhelmed", "burnout", "exhausted", "drained"],
        "sad": ["sad", "depressed", "lonely", "hopeless", "crying"],
        "angry": ["angry", "rage", "furious", "frustrated"],
        "anxious": ["anxious", "anxiety", "panic", "worried", "scared"],
        "happy": ["happy", "joy", "excited", "great"],
    }
    scores = {
        emotion: sum(len(kw.split()) for kw in kws if kw in tl)
        for emotion, kws in groups.items()
    }
    if max(scores.values()) == 0:
        return "neutral", 0.3
    top = max(scores, key=scores.get)  # type: ignore[arg-type]
    return top, min(scores[top] / 5.0, 1.0)


def detect_high_risk(text: str) -> bool:
    crisis = ["suicidal", "kill myself", "want to die", "end it all", "suicide"]
    return any(kw in text.lower() for kw in crisis)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _load_conversation(session_name: str) -> list[dict[str, str]]:
    path = os.path.join(CONVERSATION_FOLDER, f"{session_name}.json")
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
    return []


def _save_conversation(session_name: str, history: list[dict]) -> None:
    path = os.path.join(CONVERSATION_FOLDER, f"{session_name}.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    except OSError as exc:
        logger.warning("Failed to save conversation: %s", exc)


# ── Flask routes ───────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/index")
def index():
    return render_template("index.html")


@app.route("/learn_more")
def learn_more():
    return render_template("learn_more.html")


@app.route("/disclaimer")
def disclaimer():
    return render_template("disclaimer.html")


@app.route("/about")
def about():
    from datetime import datetime, timezone

    return render_template("about.html", year=datetime.now(tz=timezone.utc).year)


@app.route("/resource")
def resource():
    return render_template("resource.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/help")
def help_page():
    return render_template("help.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        files = request.files.getlist("file")
        if not files:
            return jsonify({"success": False, "error": "No files provided"}), 400
        if _indexing_service is None:
            return (
                jsonify({"success": False, "error": "RAG pipeline is unavailable"}),
                503,
            )

        saved_files: list[str] = []
        file_info: list[dict] = []
        for file in files:
            if not file or not file.filename:
                continue
            if not allowed_file(file.filename):
                logger.warning("Rejected: %s", file.filename)
                continue
            original_name = secure_filename(file.filename)
            filename = original_name
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            if os.path.exists(filepath):
                base, ext = os.path.splitext(filename)
                filename = f"{base}_{time.time_ns()}{ext}"
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            size = os.path.getsize(filepath)
            saved_files.append(filepath)
            file_info.append(
                {
                    "name": filename,
                    "original_name": original_name,
                    "path": filepath,
                    "url": url_for("uploaded_file", filename=filename),
                    "size": size,
                    "type": os.path.splitext(filename)[1].lower(),
                }
            )
            logger.info("Uploaded: %s (%d bytes)", filename, size)

        if not saved_files:
            return jsonify({"success": False, "error": "No valid files uploaded"}), 400

        document_id = _indexing_service.submit(
            saved_files, [item["original_name"] for item in file_info]
        )

        return (
            jsonify(
                {
                "success": True,
                "filepaths": saved_files,
                "filepath": saved_files[0],
                "count": len(saved_files),
                "files": file_info,
                "document_id": document_id,
                "status": "Uploading",
                "progress": 0,
                "message": "Upload received. Background indexing started.",
                }
            ),
            202,
        )
    except OSError as exc:
        logger.error("Upload error: %s", traceback.format_exc())
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/documents/<document_id>/status", methods=["GET"])
def document_status(document_id: str):
    if _indexing_service is None:
        return jsonify({"success": False, "error": "Indexing unavailable"}), 503
    status = _indexing_service.status(document_id)
    if status is None:
        return jsonify({"success": False, "error": "Document job not found"}), 404
    return jsonify({"success": True, **status})


# ── Shared helpers used by both /search and /search/stream ────────────────────
_CRISIS_MSGS: dict[str, str] = {
    "english": (
        "I can hear that you're going through something very difficult. "
        "You're not alone.\n\nIndia: +91 9152987821\nInternational: 988\n\nHelp is available 24/7."
    ),
    "hindi": "आप कुछ बहुत कठिन से गुज़र रहे हैं। आप अकेले नहीं हैं।\n\nभारत: +91 9152987821",
    "hinglish": "Aap akele nahi hain.\n\nIndia: +91 9152987821\nInternational: 988",
}
_EMOTION_EMOJI: dict[str, str] = {
    "overwhelmed": "😰",
    "sad": "😢",
    "angry": "😠",
    "anxious": "😨",
    "neutral": "😌",
    "happy": "😊",
}


def _indexed_agent(document_id: str) -> Agent | None:
    """Return an agent over already-indexed vectors; never process source files."""
    if not document_id or _indexing_service is None:
        return None
    prepared = _indexing_service.prepared_indexes(document_id)
    if prepared is None:
        return None
    indexes, checksums = prepared
    return Agent(
        indexes,
        cache_scope="response-format-v2:documents:" + "|".join(sorted(checksums)),
    )


def _used_uploaded_documents(tool_used: str, sources: list[str]) -> bool:
    if tool_used == "docs" or tool_used.startswith("docs+"):
        return True
    if tool_used == "cache":
        return any(not source.lower().startswith("web") for source in sources)
    return False


def _enrich_with_history(
    response: str,
    question: str,
    conv_history: list[dict],
    tool_used: str,
) -> str:
    """Optionally refine response using recent conversation context."""
    if not (
        conv_history
        and groq_client
        and tool_used not in ("cache", "conversational", "calculator")
    ):
        return response
    try:
        hist_text = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in conv_history[-6:]
        )
        r = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Preserve accuracy while enriching "
                        "context. Keep the answer in readable Markdown with short paragraphs, "
                        "clear section headings, and one bullet per line when useful."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Conversation so far:\n{hist_text}\n\n"
                        f"Current question: {question}\n\nBest answer so far:\n{response}\n\n"
                        "Refine using conversation context if helpful. Preserve Markdown "
                        "headings, blank lines, short paragraphs, and one bullet per line. "
                        "For substantial answers, retain or add a concise Summary section. "
                        "If already complete, return it unchanged."
                    ),
                },
            ],
            temperature=0.4,
            max_tokens=1000,
        )
        return r.choices[0].message.content or response
    except (OSError, RuntimeError) as exc:
        logger.warning("History enrichment failed: %s", exc)
        return response


# ── Main search endpoint ───────────────────────────────────────────────────────
@app.route("/search", methods=["POST"])
def search():
    try:
        t0 = time.time()
        data = request.get_json(force=True)
        question: str = data.get("query") or data.get("question", "")
        session_name: str = data.get("session_name", "default_session")
        document_id: str = data.get("document_id", "")

        if not question:
            return jsonify({"success": False, "error": "Query required"}), 400

        logger.info("[search] Q=%r session=%s", question[:100], session_name)

        lang = detect_language(question)
        emotion, confidence = detect_emotion(question)
        emotion_emoji = _EMOTION_EMOJI.get(emotion, "💭")

        if detect_high_risk(question):
            return jsonify(
                {
                    "success": True,
                    "emotion_detected": emotion,
                    "language": lang,
                    "response": _CRISIS_MSGS.get(lang, _CRISIS_MSGS["english"]),
                }
            )

        conv_history = _load_conversation(session_name)
        indexed_agent = _indexed_agent(document_id)
        if document_id and indexed_agent is None:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Document indexing is not complete",
                    }
                ),
                409,
            )

        # Branch A — agentic RAG
        if indexed_agent is not None:
            try:
                agent_response, sources, tool_used = indexed_agent.run(question)
                agent_response = _enrich_with_history(
                    agent_response, question, conv_history, tool_used
                )

                if sources:
                    agent_response += "\n\nSources: " + ", ".join(
                        dict.fromkeys(sources)
                    )

                conv_history.append({"role": "user", "content": question})
                conv_history.append({"role": "assistant", "content": agent_response})
                _save_conversation(session_name, conv_history)

                return jsonify(
                    {
                        "success": True,
                        "emotion_detected": emotion,
                        "emotion_emoji": emotion_emoji,
                        "emotion_confidence": round(confidence, 2),
                        "language": lang,
                        "model_used": "llama-3.3-70b-versatile",
                        "rag_used": _used_uploaded_documents(tool_used, sources),
                        "tool_used": tool_used,
                        "sources": sources,
                        "response": agent_response,
                        "time": round(time.time() - t0, 2),
                    }
                )
            except (OSError, RuntimeError, ValueError) as exc:
                logger.error("Agentic RAG error: %s", traceback.format_exc())
                logger.warning("Falling back to direct Groq: %s", exc)

        # Branch B — direct Groq
        if not groq_client:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Groq client not available.",
                        "response": "Service unavailable.",
                    }
                ),
                500,
            )

        user_message = question
        if emotion != "neutral" and confidence > 0.5:
            user_message += (
                f"\n\n[Detected emotion: {emotion}, confidence: {confidence:.2f}]"
            )

        messages: list[ChatCompletionMessageParam] = cast(
            list[ChatCompletionMessageParam],
            [{"role": "system", "content": prompt.prompts}],
        )
        for msg in conv_history[-10:]:
            messages.append(
                cast(
                    ChatCompletionMessageParam,
                    {"role": msg["role"], "content": msg["content"]},
                )
            )
        messages.append(
            cast(ChatCompletionMessageParam, {"role": "user", "content": user_message})
        )

        try:
            resp = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.7,
                top_p=0.9,
                max_tokens=2000,
            )
            assistant_reply: str = resp.choices[0].message.content or ""
        except (OSError, RuntimeError, ValueError) as exc:
            logger.error("Groq error: %s", exc)
            return (
                jsonify(
                    {
                        "success": False,
                        "error": str(exc),
                        "response": "Response generation failed.",
                    }
                ),
                500,
            )

        conv_history.append({"role": "user", "content": question})
        conv_history.append({"role": "assistant", "content": assistant_reply})
        _save_conversation(session_name, conv_history)

        return jsonify(
            {
                "success": True,
                "emotion_detected": emotion,
                "emotion_emoji": emotion_emoji,
                "emotion_confidence": round(confidence, 2),
                "language": lang,
                "model_used": "llama-3.3-70b-versatile",
                "rag_used": False,
                "tool_used": "llm",
                "sources": [],
                "response": assistant_reply,
                "time": round(time.time() - t0, 2),
            }
        )

    except (OSError, ValueError) as exc:
        logger.error("Search error: %s", traceback.format_exc())
        return (
            jsonify(
                {"success": False, "error": str(exc), "response": "An error occurred."}
            ),
            500,
        )


# ── Streaming search endpoint ─────────────────────────────────────────────────
@app.route("/search/stream", methods=["GET"])
def search_stream():
    """SSE endpoint with native LLM streaming and batched completed tool results."""
    question = request.args.get("q", "").strip()
    session_name = request.args.get("session_name", "default_session")
    document_id = request.args.get("document_id", "").strip()

    def _sse(payload: dict) -> str:
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    def generate() -> Generator[str, None, None]:
        if not question:
            yield _sse({"type": "error", "message": "Query required"})
            return

        lang = detect_language(question)
        emotion, conf = detect_emotion(question)
        emotion_emoji = _EMOTION_EMOJI.get(emotion, "💭")

        if detect_high_risk(question):
            msg = _CRISIS_MSGS.get(lang, _CRISIS_MSGS["english"])
            yield _sse(
                {
                    "type": "meta",
                    "emotion_detected": emotion,
                    "emotion_emoji": emotion_emoji,
                    "language": lang,
                    "tool_used": "crisis",
                }
            )
            yield _sse({"type": "chunk", "text": msg})
            yield _sse({"type": "done", "sources": []})
            return

        indexed_agent = _indexed_agent(document_id)
        if document_id and indexed_agent is None:
            yield _sse(
                {"type": "error", "message": "Document indexing is not complete"}
            )
            return
        full_response = ""
        sources: list[str] = []
        tool_used = "llm"
        response_already_streamed = False

        try:
            if indexed_agent is not None:
                full_response, sources, tool_used = indexed_agent.run(question)

                conv_history = _load_conversation(session_name)
                full_response = _enrich_with_history(
                    full_response, question, conv_history, tool_used
                )

            else:
                if not groq_client:
                    yield _sse(
                        {"type": "error", "message": "Groq client not available"}
                    )
                    return

                conv_history = _load_conversation(session_name)
                user_msg = question
                if emotion != "neutral" and conf > 0.5:
                    user_msg += (
                        f"\n\n[Detected emotion: {emotion}, confidence: {conf:.2f}]"
                    )

                chat_messages: list[ChatCompletionMessageParam] = cast(
                    list[ChatCompletionMessageParam],
                    [{"role": "system", "content": prompt.prompts}],
                )
                for hist_msg in conv_history[-10:]:
                    chat_messages.append(
                        cast(
                            ChatCompletionMessageParam,
                            {"role": hist_msg["role"], "content": hist_msg["content"]},
                        )
                    )
                chat_messages.append(
                    cast(
                        ChatCompletionMessageParam,
                        {"role": "user", "content": user_msg},
                    )
                )

                # Native provider streaming improves time-to-first-response. The
                # browser buffers these small deltas into readable visual chunks.
                yield _sse(
                    {
                        "type": "meta",
                        "emotion_detected": emotion,
                        "emotion_emoji": emotion_emoji,
                        "emotion_confidence": round(conf, 2),
                        "language": lang,
                        "tool_used": "llm",
                        "rag_used": False,
                    }
                )
                response_stream = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=chat_messages,
                    temperature=0.7,
                    top_p=0.9,
                    max_tokens=2000,
                    stream=True,
                )
                for completion_chunk in response_stream:
                    chunk_text = completion_chunk.choices[0].delta.content or ""
                    if chunk_text:
                        full_response += chunk_text
                        yield _sse({"type": "chunk", "text": chunk_text})
                response_already_streamed = True
                tool_used = "llm"

        except (OSError, RuntimeError, ValueError) as exc:
            logger.error("Stream generation error: %s", traceback.format_exc())
            yield _sse({"type": "error", "message": str(exc)})
            return

        if sources:
            full_response += "\n\nSources: " + ", ".join(dict.fromkeys(sources))

        conv_history = _load_conversation(session_name)
        conv_history.append({"role": "user", "content": question})
        conv_history.append({"role": "assistant", "content": full_response})
        _save_conversation(session_name, conv_history)

        if not response_already_streamed:
            yield _sse(
                {
                    "type": "meta",
                    "emotion_detected": emotion,
                    "emotion_emoji": emotion_emoji,
                    "emotion_confidence": round(conf, 2),
                    "language": lang,
                    "tool_used": tool_used,
                    "rag_used": _used_uploaded_documents(tool_used, sources),
                }
            )
            # Tool/RAG calls return a completed answer. The browser converts this
            # frame into a fast progressive token reveal without delaying the server.
            yield _sse({"type": "chunk", "text": full_response})

        yield _sse({"type": "done", "sources": sources})

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ── Conversation management ────────────────────────────────────────────────────
@app.route("/conversation/history", methods=["POST"])
def get_conversation_history():
    try:
        data = request.get_json(force=True)
        session_name = data.get("session_name", "default_session")
        history = _load_conversation(session_name)
        return jsonify({"success": True, "history": history, "count": len(history)})
    except (OSError, ValueError) as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/conversation/clear", methods=["POST"])
def clear_conversation_history():
    try:
        data = request.get_json(force=True)
        session_name = data.get("session_name", "default_session")
        path = os.path.join(CONVERSATION_FOLDER, f"{session_name}.json")
        if os.path.exists(path):
            os.remove(path)
        return jsonify({"success": True, "message": "Conversation history cleared"})
    except OSError as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


# ── Email ──────────────────────────────────────────────────────────────────────
@app.route("/send_email", methods=["POST"])
def send_email():
    try:
        if not (SENDER_EMAIL and SENDER_PASSWORD):
            raise ValueError("Email configuration missing")
        name = request.form["name"]
        email = request.form["email"]
        message = request.form["message"]
        msg = MIMEMultipart()
        msg["From"] = SENDER_EMAIL
        msg["To"] = RECEIVER_EMAIL
        msg["Subject"] = f"Contact Form: {name}"
        msg.attach(
            MIMEText(f"Name: {name}\nEmail: {email}\n\nMessage:\n{message}", "plain")
        )
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
        flash("Message sent successfully!", "success")
    except (OSError, smtplib.SMTPException, ValueError) as exc:
        logger.error("Email error: %s", exc)
        flash(f"Error: {exc}", "error")
    return redirect(url_for("contact"))


# ── Static / health ────────────────────────────────────────────────────────────
@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "online",
            "groq_available": groq_client is not None,
            "groq_sdk_installed": GROQ_AVAILABLE,
            "api_key_set": bool(GROQ_API_KEY),
            "rag_available": RAG_AVAILABLE,
            "agentic_rag_available": AGENTIC_RAG_AVAILABLE,
            "rag_settings_initialised": _rag_settings_initialised,
            "supported_formats": sorted(ALLOWED_EXTENSIONS),
        }
    )


if __name__ == "__main__":
    print("CentrixSupport — Mental Health AI Chatbot (Agentic RAG)")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), debug=False)
