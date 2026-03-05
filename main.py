"""
CandidateProfiler — FastAPI Main App
Serves the Jinja2 frontend and exposes API endpoints for resume upload, chat, and payload generation.
"""

import logging
import traceback
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models import ChatMessage, ResumeSummary, CandidatePayload

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# In-memory session storage (replace with Redis/DB in production)
sessions: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup."""
    logger.info("=" * 60)
    logger.info("CandidateProfiler Starting...")
    logger.info("=" * 60)
    logger.info("✅ Server ready at http://localhost:8000")
    logger.info("=" * 60)
    yield
    logger.info("Server shutting down...")


app = FastAPI(
    title="CandidateProfiler",
    description="AI-powered candidate profiling and career guidance",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ============================================================================
# Pages
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main profiler page."""
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "chat_history": [],
        "resume_summary": None,
        "resume_raw_text": None,
        "resume_uploaded": False,
        "payload": None,
    }
    return templates.TemplateResponse("index.html", {
        "request": request,
        "session_id": session_id,
    })


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "candidate-profiler", "version": "1.0.0"}


# ============================================================================
# API Endpoints
# ============================================================================

@app.post("/api/upload-resume")
async def upload_resume(file: UploadFile = File(...), session_id: str = ""):
    """Upload and parse a resume (PDF or DOCX). Fast mode: no LLM call."""
    from resume_parser import parse_resume

    if session_id not in sessions:
        sessions[session_id] = {
            "chat_history": [], "resume_summary": None,
            "resume_raw_text": None, "resume_uploaded": False, "payload": None,
        }

    # Validate file type
    filename = file.filename or "resume.pdf"
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if ext not in ("pdf", "docx", "doc"):
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported.")

    try:
        content = await file.read()
        if len(content) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=400, detail="File too large. Max 10MB.")

        raw_text, preview = parse_resume(content, filename)

        # Store in session
        sessions[session_id]["resume_summary"] = preview  # dict, not Pydantic
        sessions[session_id]["resume_raw_text"] = raw_text
        sessions[session_id]["resume_uploaded"] = True

        return {
            "success": True,
            "filename": filename,
            "summary": preview,  # dict with name, email, skills, etc.
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Resume upload failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to parse resume: {str(e)}")


class ChatRequest(BaseModel):
    session_id: str
    message: str


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Send a message and get the agent's response."""
    import asyncio

    logger.info(f"CHAT REQUEST: session={request.session_id}, message='{request.message[:50] if request.message else ''}'")

    session = sessions.get(request.session_id)
    if not session:
        session = {
            "chat_history": [], "resume_summary": None,
            "resume_raw_text": None, "resume_uploaded": False, "payload": None,
        }
        sessions[request.session_id] = session

    # Add user message to history
    if request.message:
        session["chat_history"].append(ChatMessage(
            role="user",
            content=request.message,
        ))

    # ── FAST PATH: instant first message (works for BOTH skip and resume) ──
    is_first_turn = (not request.message and len(session["chat_history"]) == 0)
    if is_first_turn:
        has_resume = bool(session.get("resume_raw_text"))

        if has_resume:
            # Resume uploaded: personalized greeting based on regex-extracted preview
            preview = session.get("resume_summary", {})
            skills = preview.get("skills", [])
            skill_text = f" I can see skills like {', '.join(skills[:3])} on your resume." if skills else ""
            first_msg = f"Hey! Thanks for sharing your resume.{skill_text} Let's find your perfect next role.|||Which of these best describes you right now?"
        else:
            # No resume: generic greeting
            first_msg = "Hey there! I'm StudojoProfiler, your career buddy. Let's find your perfect next role.|||Which of these best describes you right now?"

        first_mcq = {
            "question": "Which of these best describes you right now?",
            "options": [
                {"label": "A", "text": "I'm a student, not graduating soon"},
                {"label": "B", "text": "I'm a student, graduating within 6 months"},
                {"label": "C", "text": "Recent graduate, 0-2 years of experience"},
                {"label": "D", "text": "Experienced professional, 3+ years"},
                {"label": "E", "text": "I'm switching careers or exploring new fields"},
                {"label": "F", "text": "Other"},
            ],
            "allow_multiple": False,
        }
        session["chat_history"].append(ChatMessage(role="assistant", content=first_msg))
        logger.info("FAST PATH: returning instant first message")
        return {
            "message": first_msg,
            "state": "MCQ",
            "mcq": first_mcq,
            "text_input": False,
            "is_complete": False,
            "questions_asked": 1,
        }

    # ── NORMAL PATH: call LLM with timeout ──
    try:
        from profiler_agent import get_agent_response
        logger.info("LLM PATH: calling get_agent_response...")
        response = await asyncio.wait_for(
            asyncio.to_thread(
                get_agent_response,
                session["chat_history"],
                session.get("resume_summary"),
                session.get("resume_raw_text"),
            ),
            timeout=30.0,  # 30 second timeout
        )
        logger.info(f"LLM PATH: got response, state={response.current_state}")

        session["chat_history"].append(ChatMessage(
            role="assistant",
            content=response.message,
            mcq=response.mcq,
        ))

        return {
            "message": response.message,
            "state": response.current_state,
            "mcq": response.mcq.model_dump() if response.mcq else None,
            "text_input": response.text_input,
            "is_complete": response.is_complete,
            "questions_asked": len([m for m in session["chat_history"] if m.role == "assistant"]),
        }

    except asyncio.TimeoutError:
        logger.error("Chat timeout: LLM call took > 30 seconds")
        return {
            "message": "That took too long, let me try again. What were you saying?",
            "state": "MCQ",
            "mcq": None,
            "text_input": False,
            "is_complete": False,
            "questions_asked": len([m for m in session["chat_history"] if m.role == "assistant"]),
        }
    except Exception as e:
        logger.error(f"Chat error: {e}\n{traceback.format_exc()}")
        return {
            "message": "Sorry, I hit a snag. Could you try again?",
            "state": "MCQ",
            "mcq": None,
            "text_input": False,
            "is_complete": False,
            "questions_asked": 0,
        }


class PayloadRequest(BaseModel):
    session_id: str


@app.post("/api/generate-payload")
async def generate_payload(request: PayloadRequest):
    """Generate the final candidate profile payload."""
    import asyncio
    from profiler_agent import generate_final_payload

    session = sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if len(session["chat_history"]) < 4:
        raise HTTPException(status_code=400, detail="Not enough conversation data to generate a profile")

    try:
        payload = await asyncio.to_thread(
            generate_final_payload,
            session["chat_history"],
            session.get("resume_summary"),
            session.get("resume_raw_text"),
            session.get("resume_uploaded", False),
        )

        session["payload"] = payload
        return payload.model_dump()

    except Exception as e:
        logger.error(f"Payload generation failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to generate payload: {str(e)}")


@app.post("/api/skip-resume")
async def skip_resume(request: PayloadRequest):
    """Mark session as 'no resume uploaded' and start chat."""
    session_id = request.session_id
    if session_id not in sessions:
        sessions[session_id] = {
            "chat_history": [], "resume_summary": None,
            "resume_raw_text": None, "resume_uploaded": False, "payload": None,
        }

    sessions[session_id]["resume_uploaded"] = False
    return {"success": True}


# Run directly with: python main.py
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
