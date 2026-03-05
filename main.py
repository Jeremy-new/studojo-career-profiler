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


# ============================================================================
# Server-Side Question Flow (ZERO LLM calls during chat)
# The gpt-5-mini reasoning model takes 60-120s per call.
# So we hardcode all 8 questions and only call the LLM once for final payload.
# ============================================================================

QUESTION_FLOW = [
    {
        "id": "stage",
        "ack": None,  # First question has no ack
        "message": "Which of these best describes you right now?",
        "mcq": {
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
        },
        "text_input": False,
    },
    {
        "id": "job_type",
        "ack": "Got it!",
        "message": "Are you looking for an internship or a full-time role?",
        "mcq": {
            "question": "Are you looking for an internship or a full-time role?",
            "options": [
                {"label": "A", "text": "Full-time job"},
                {"label": "B", "text": "Internship (3-6 months)"},
                {"label": "C", "text": "Part-time or freelance"},
                {"label": "D", "text": "Open to all options"},
                {"label": "E", "text": "Other"},
            ],
            "allow_multiple": False,
        },
        "text_input": False,
    },
    {
        "id": "location",
        "ack": "Noted.",
        "message": "Which cities or regions would you prefer to work in?",
        "mcq": {
            "question": "Which cities or regions would you prefer to work in?",
            "options": [
                {"label": "A", "text": "Bengaluru"},
                {"label": "B", "text": "Mumbai"},
                {"label": "C", "text": "Delhi NCR"},
                {"label": "D", "text": "Hyderabad"},
                {"label": "E", "text": "Pune"},
                {"label": "F", "text": "Chennai"},
                {"label": "G", "text": "Remote"},
                {"label": "H", "text": "Other"},
            ],
            "allow_multiple": True,
        },
        "text_input": False,
    },
    {
        "id": "company_stage",
        "ack": "Great choices.",
        "message": "What type of company do you want to join?",
        "mcq": {
            "question": "What type of company do you want to join?",
            "options": [
                {"label": "A", "text": "Early-stage startup (under 50 people)"},
                {"label": "B", "text": "Growth-stage startup (50-250 people)"},
                {"label": "C", "text": "Large company or enterprise (250+)"},
                {"label": "D", "text": "MNC or global corporation"},
                {"label": "E", "text": "No preference"},
                {"label": "F", "text": "Other"},
            ],
            "allow_multiple": False,
        },
        "text_input": False,
    },
    {
        "id": "industry",
        "ack": "Makes sense.",
        "message": "Which industries excite you most?",
        "mcq": {
            "question": "Which industries excite you most?",
            "options": [
                {"label": "A", "text": "Fintech / Payments"},
                {"label": "B", "text": "Edtech / Education"},
                {"label": "C", "text": "Healthcare / Healthtech"},
                {"label": "D", "text": "E-commerce / D2C"},
                {"label": "E", "text": "SaaS / Enterprise Software"},
                {"label": "F", "text": "AI / Machine Learning"},
                {"label": "G", "text": "Media / Content"},
                {"label": "H", "text": "Other"},
            ],
            "allow_multiple": True,
        },
        "text_input": False,
    },
    {
        "id": "salary",
        "ack": "Good to know.",
        "message": "What's your expected annual salary range (CTC)?",
        "mcq": None,
        "text_input": True,
    },
    {
        "id": "role_focus",
        "ack": "Thanks, noted.",
        "message": "What kind of work do you enjoy most?",
        "mcq": {
            "question": "What kind of work do you enjoy most?",
            "options": [
                {"label": "A", "text": "Building product/features"},
                {"label": "B", "text": "Growth and marketing"},
                {"label": "C", "text": "Strategy and business development"},
                {"label": "D", "text": "Analyzing data and insights"},
                {"label": "E", "text": "Stakeholder management and partnerships"},
                {"label": "F", "text": "Managing teams and people"},
                {"label": "G", "text": "Other"},
            ],
            "allow_multiple": False,
        },
        "text_input": False,
    },
    {
        "id": "skills",
        "ack": "Great pick.",
        "message": "Which skills do you want to use or develop in your next role?",
        "mcq": {
            "question": "Which skills do you want to use or develop?",
            "options": [
                {"label": "A", "text": "Data analysis and SQL"},
                {"label": "B", "text": "Product management"},
                {"label": "C", "text": "Marketing and growth"},
                {"label": "D", "text": "Programming and engineering"},
                {"label": "E", "text": "Design and UX"},
                {"label": "F", "text": "Communication and leadership"},
                {"label": "G", "text": "Other"},
            ],
            "allow_multiple": True,
        },
        "text_input": False,
    },
]


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Server-side question flow. ZERO LLM calls. Instant responses."""

    logger.info(f"CHAT REQUEST: session={request.session_id}, message='{request.message[:50] if request.message else ''}'")

    session = sessions.get(request.session_id)
    if not session:
        session = {
            "chat_history": [], "resume_summary": None,
            "resume_raw_text": None, "resume_uploaded": False, "payload": None,
            "question_index": 0,
        }
        sessions[request.session_id] = session

    # Ensure question_index exists (for legacy sessions)
    if "question_index" not in session:
        session["question_index"] = 0

    qi = session["question_index"]

    # ── FIRST TURN: greeting + Q1 ──
    if not request.message and qi == 0:
        has_resume = bool(session.get("resume_raw_text"))
        q = QUESTION_FLOW[0]

        if has_resume:
            preview = session.get("resume_summary", {})
            skills = preview.get("skills", [])
            skill_text = f" I can see skills like {', '.join(skills[:3])} on your resume." if skills else ""
            greeting = f"Hey! Thanks for sharing your resume.{skill_text} Let's find your perfect next role."
        else:
            greeting = "Hey there! I'm StudojoProfiler, your career buddy. Let's find your perfect next role."

        msg = f"{greeting}|||{q['message']}"
        session["chat_history"].append(ChatMessage(role="assistant", content=msg))
        logger.info("FAST PATH: Q1 served instantly")

        return {
            "message": msg,
            "state": "MCQ",
            "mcq": q["mcq"],
            "text_input": q["text_input"],
            "is_complete": False,
            "questions_asked": 1,
        }

    # ── RECORD USER ANSWER ──
    if request.message:
        session["chat_history"].append(ChatMessage(role="user", content=request.message))

    # ── ADVANCE TO NEXT QUESTION ──
    session["question_index"] = qi + 1
    next_qi = session["question_index"]

    # ── ALL QUESTIONS DONE → COMPLETE ──
    if next_qi >= len(QUESTION_FLOW):
        done_msg = "Thanks for answering all my questions! I have everything I need. Generating your career profile now..."
        session["chat_history"].append(ChatMessage(role="assistant", content=done_msg))
        logger.info(f"COMPLETE: all {len(QUESTION_FLOW)} questions answered")
        return {
            "message": done_msg,
            "state": "PAYLOAD_READY",
            "mcq": None,
            "text_input": False,
            "is_complete": True,
            "questions_asked": next_qi,
        }

    # ── SERVE NEXT QUESTION INSTANTLY ──
    q = QUESTION_FLOW[next_qi]
    ack = q["ack"] or ""
    msg = f"{ack}|||{q['message']}" if ack else q["message"]

    session["chat_history"].append(ChatMessage(role="assistant", content=msg))
    logger.info(f"SERVED Q{next_qi + 1}/{len(QUESTION_FLOW)}: {q['id']}")

    return {
        "message": msg,
        "state": "MCQ",
        "mcq": q["mcq"],
        "text_input": q["text_input"],
        "is_complete": False,
        "questions_asked": next_qi + 1,
    }


class PayloadRequest(BaseModel):
    session_id: str


@app.post("/api/generate-payload")
async def generate_payload(request: PayloadRequest):
    """Generate the final candidate profile payload and save as .md to Outputs/."""
    import asyncio
    import json
    from datetime import datetime
    from profiler_agent import generate_final_payload

    session = sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if len(session["chat_history"]) < 4:
        raise HTTPException(status_code=400, detail="Not enough conversation data to generate a profile")

    try:
        payload = await asyncio.wait_for(
            asyncio.to_thread(
                generate_final_payload,
                session["chat_history"],
                session.get("resume_summary"),
                session.get("resume_raw_text"),
                session.get("resume_uploaded", False),
            ),
            timeout=60.0,  # Payload generation can take longer
        )

        session["payload"] = payload
        payload_dict = payload.model_dump()

        # ── Save as .md to Outputs/ folder ──
        outputs_dir = os.path.join(os.path.dirname(__file__), "Outputs")
        os.makedirs(outputs_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        candidate_name = (payload_dict.get("personal_info", {}).get("name") or "unknown").replace(" ", "_")
        filename = f"{timestamp}_{candidate_name}.md"
        filepath = os.path.join(outputs_dir, filename)

        # Build the .md content
        md_content = f"""---
candidate_id: {payload_dict.get('candidate_id', 'N/A')}
timestamp: {payload_dict.get('timestamp', 'N/A')}
resume_uploaded: {payload_dict.get('session_metadata', {}).get('resume_uploaded', False)}
questions_answered: {payload_dict.get('session_metadata', {}).get('questions_answered', 0)}
confidence_score: {payload_dict.get('session_metadata', {}).get('confidence_score', 0)}
---

# Candidate Profile: {payload_dict.get('personal_info', {}).get('name') or 'Unknown'}

## Profile Summary
{payload_dict.get('profile_summary', 'N/A')}

## Personal Info
- **Name:** {payload_dict.get('personal_info', {}).get('name') or 'N/A'}
- **Email:** {payload_dict.get('personal_info', {}).get('email') or 'N/A'}
- **Skills:** {', '.join(payload_dict.get('personal_info', {}).get('skills_detected', []))}

## Preferences
- **Locations:** {', '.join(payload_dict.get('preferences', {}).get('locations', []))}
- **Work Mode:** {payload_dict.get('preferences', {}).get('work_mode', 'N/A')}
- **Company Stage:** {payload_dict.get('preferences', {}).get('company_stage', 'N/A')}
- **Industries:** {', '.join(payload_dict.get('preferences', {}).get('industry_interests', []))}
- **Salary:** {payload_dict.get('preferences', {}).get('salary_expectations', {}).get('currency', 'INR')} {payload_dict.get('preferences', {}).get('salary_expectations', {}).get('min_annual_ctc', 0):,} - {payload_dict.get('preferences', {}).get('salary_expectations', {}).get('max_annual_ctc', 0):,}

## Career Analysis
- **Primary Cluster:** {payload_dict.get('career_analysis', {}).get('primary_cluster', 'N/A')}
- **Secondary Cluster:** {payload_dict.get('career_analysis', {}).get('secondary_cluster', 'N/A')}

### Recommended Roles
"""
        for role in payload_dict.get("career_analysis", {}).get("recommended_roles", []):
            md_content += f"| {role.get('title', 'N/A')} | {role.get('seniority', 'N/A')} | {role.get('fit_score', 0):.0%} | {role.get('reasoning', '')} |\n"

        md_content += f"""
### Transition Paths
"""
        for path in payload_dict.get("career_analysis", {}).get("transition_paths", []):
            md_content += f"- {path}\n"

        md_content += f"""
---

## Full JSON Payload

```json
{json.dumps(payload_dict, indent=2)}
```
"""

        with open(filepath, "w") as f:
            f.write(md_content)

        logger.info(f"Payload saved to: {filepath}")
        return {"payload": payload_dict, "saved_to": filename}

    except asyncio.TimeoutError:
        logger.error("Payload generation timed out (>60s)")
        raise HTTPException(status_code=504, detail="Payload generation timed out. Try again.")
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
