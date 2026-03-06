"""
CandidateProfiler — FastAPI Main App
Serves the Jinja2 frontend and exposes API endpoints for resume upload, chat, and payload generation.
"""

import os
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
# Career Ontology — Dynamic Question Generation
# ============================================================================

from career_ontology import CAREER_ONTOLOGY, get_all_clusters, get_specializations


# Static question templates
STATIC_QUESTIONS = {
    "stage": {
        "ack": None,
        "message": "Which of these best describes you right now?",
        "mcq": {
            "question": "Which of these best describes you right now?",
            "options": [
                {"label": "A", "text": "I'm a student, not graduating soon"},
                {"label": "B", "text": "I'm a student, graduating within 6 months"},
                {"label": "C", "text": "Recent graduate (0-2 years experience)"},
                {"label": "D", "text": "Experienced professional (3+ years)"},
                {"label": "E", "text": "Switching careers or exploring new fields"},
                {"label": "F", "text": "Other"},
            ],
            "allow_multiple": False,
        },
        "text_input": False,
    },
    "job_type": {
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
    # domain + specialization are built dynamically from ontology
    "location": {
        "ack": "Interesting!",
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
                {"label": "G", "text": "Kolkata"},
                {"label": "H", "text": "Remote"},
                {"label": "I", "text": "International"},
                {"label": "J", "text": "Other"},
            ],
            "allow_multiple": True,
        },
        "text_input": False,
    },
    "work_style": {
        "ack": "Good choices.",
        "message": "What's your preferred work style?",
        "mcq": {
            "question": "What's your preferred work style?",
            "options": [
                {"label": "A", "text": "Fully remote"},
                {"label": "B", "text": "Hybrid (mix of office and remote)"},
                {"label": "C", "text": "Fully on-site / in office"},
                {"label": "D", "text": "Flexible, no strong preference"},
                {"label": "E", "text": "Other"},
            ],
            "allow_multiple": False,
        },
        "text_input": False,
    },
    "company_stage": {
        "ack": "Noted.",
        "message": "What type of company appeals to you the most?",
        "mcq": {
            "question": "What type of company appeals to you?",
            "options": [
                {"label": "A", "text": "Early-stage startup (seed / under 50 people)"},
                {"label": "B", "text": "Growth-stage startup (50-500 people)"},
                {"label": "C", "text": "Mid-size company (500-2000)"},
                {"label": "D", "text": "Large enterprise or MNC (2000+)"},
                {"label": "E", "text": "Government / Public sector"},
                {"label": "F", "text": "No preference"},
                {"label": "G", "text": "Other"},
            ],
            "allow_multiple": False,
        },
        "text_input": False,
    },
    "industry": {
        "ack": "Makes sense.",
        "message": "Which industries excite you the most?",
        "mcq": {
            "question": "Which industries excite you the most?",
            "options": [
                {"label": "A", "text": "Fintech / Payments / Banking"},
                {"label": "B", "text": "Edtech / Education"},
                {"label": "C", "text": "Healthcare / Healthtech"},
                {"label": "D", "text": "E-commerce / D2C / Retail"},
                {"label": "E", "text": "SaaS / Enterprise Software"},
                {"label": "F", "text": "AI / Machine Learning / Deep Tech"},
                {"label": "G", "text": "Media / Content / Gaming"},
                {"label": "H", "text": "Consulting / Professional Services"},
                {"label": "I", "text": "Manufacturing / Automotive"},
                {"label": "J", "text": "Other"},
            ],
            "allow_multiple": True,
        },
        "text_input": False,
    },
    "salary": {
        "ack": "Good to know.",
        "message": "What's your expected annual salary or CTC range? (e.g. ₹4-6 LPA, ₹15-20 LPA)",
        "mcq": None,
        "text_input": True,
    },
    "role_focus": {
        "ack": "Thanks, noted.",
        "message": "What kind of day-to-day work do you enjoy the most?",
        "mcq": {
            "question": "What kind of day-to-day work do you enjoy most?",
            "options": [
                {"label": "A", "text": "Building products and writing code"},
                {"label": "B", "text": "Analyzing data and finding insights"},
                {"label": "C", "text": "Designing user experiences and visuals"},
                {"label": "D", "text": "Marketing, growth, and customer acquisition"},
                {"label": "E", "text": "Strategy, planning, and business development"},
                {"label": "F", "text": "Managing teams and stakeholders"},
                {"label": "G", "text": "Research, writing, and content creation"},
                {"label": "H", "text": "Other"},
            ],
            "allow_multiple": True,
        },
        "text_input": False,
    },
    "skills": {
        "ack": "Great choice!",
        "message": "Which skills do you want to actively use or develop in your next role?",
        "mcq": {
            "question": "Which skills do you want to use or grow?",
            "options": [
                {"label": "A", "text": "Python / JavaScript / Programming"},
                {"label": "B", "text": "Data analysis, SQL, Excel"},
                {"label": "C", "text": "Product management / Roadmapping"},
                {"label": "D", "text": "UI/UX Design / Figma"},
                {"label": "E", "text": "Digital marketing / SEO / Ads"},
                {"label": "F", "text": "Communication and public speaking"},
                {"label": "G", "text": "Leadership and people management"},
                {"label": "H", "text": "Machine learning / AI / Deep learning"},
                {"label": "I", "text": "Financial analysis / Modeling"},
                {"label": "J", "text": "Other"},
            ],
            "allow_multiple": True,
        },
        "text_input": False,
    },
    "timeline": {
        "ack": "Got it!",
        "message": "When are you looking to start your next role?",
        "mcq": {
            "question": "When do you want to start?",
            "options": [
                {"label": "A", "text": "Immediately (within 1 month)"},
                {"label": "B", "text": "In 1-3 months"},
                {"label": "C", "text": "In 3-6 months"},
                {"label": "D", "text": "6+ months (just exploring)"},
                {"label": "E", "text": "Other"},
            ],
            "allow_multiple": False,
        },
        "text_input": False,
    },
    "education_level": {
        "ack": "Understood.",
        "message": "What is your highest level of education?",
        "mcq": {
            "question": "What is your highest level of education?",
            "options": [
                {"label": "A", "text": "High School / Diploma"},
                {"label": "B", "text": "Bachelor's Degree"},
                {"label": "C", "text": "Master's Degree"},
                {"label": "D", "text": "PhD / Doctorate"},
                {"label": "E", "text": "Self-taught / Bootcamp"},
                {"label": "F", "text": "Other"},
            ],
            "allow_multiple": False,
        },
        "text_input": False,
    },
    "years_experience": {
        "ack": "Got it.",
        "message": "Roughly how many years of total professional experience do you have?",
        "mcq": {
            "question": "Years of experience?",
            "options": [
                {"label": "A", "text": "0-1 years (Entry level)"},
                {"label": "B", "text": "1-3 years (Junior)"},
                {"label": "C", "text": "3-5 years (Mid-level)"},
                {"label": "D", "text": "5-10 years (Senior)"},
                {"label": "E", "text": "10+ years (Staff / Lead)"},
            ],
            "allow_multiple": False,
        },
        "text_input": False,
    },
}

# 12 questions in order. "domain" and "specialization" are dynamically generated.
QUESTION_ORDER = [
    "stage", "job_type", "domain", "specialization", "location",
    "work_style", "company_stage", "industry", "salary",
    "role_focus", "skills", "timeline",
]


def _build_domain_question() -> dict:
    """Dynamically build domain Q from career ontology clusters."""
    clusters = get_all_clusters()
    options = []
    for i, cluster in enumerate(clusters[:12]):
        options.append({"label": chr(65 + i), "text": cluster})
    options.append({"label": chr(65 + len(options)), "text": "Other"})
    return {
        "ack": "Nice.",
        "message": "What broad career domain excites you the most? Pick 1-3.",
        "mcq": {
            "question": "What broad career domain excites you the most?",
            "options": options,
            "allow_multiple": True,
        },
        "text_input": False,
    }


def _build_specialization_question(selected_domains: list[str]) -> dict:
    """Dynamically build specialization Q from the selected domain(s)."""
    specs = []
    for domain in selected_domains:
        for cluster_name, specializations in CAREER_ONTOLOGY.items():
            if domain.lower() in cluster_name.lower() or cluster_name.lower() in domain.lower():
                specs.extend(list(specializations.keys()))

    # Deduplicate and cap at 10
    seen = set()
    unique_specs = []
    for s in specs:
        if s not in seen:
            seen.add(s)
            unique_specs.append(s)
    unique_specs = unique_specs[:10]

    if not unique_specs:
        # Fallback: show popular specializations
        for _, specializations in list(CAREER_ONTOLOGY.items())[:5]:
            unique_specs.extend(list(specializations.keys())[:2])
        unique_specs = unique_specs[:10]

    options = []
    for i, spec in enumerate(unique_specs):
        options.append({"label": chr(65 + i), "text": spec})
    options.append({"label": chr(65 + len(options)), "text": "Other"})

    domain_names = ", ".join(selected_domains[:2])
    return {
        "ack": "Great choices!",
        "message": f"Within {domain_names}, which specializations interest you?",
        "mcq": {
            "question": f"Which specialization in {domain_names} interests you most?",
            "options": options,
            "allow_multiple": True,
        },
        "text_input": False,
    }


def _get_question(q_id: str, session: dict) -> dict:
    """Get a question by ID, building it dynamically if needed."""
    if q_id == "domain":
        return _build_domain_question()
    elif q_id == "specialization":
        answers = session.get("answers", {})
        domains = answers.get("domain", [])
        if isinstance(domains, str):
            domains = [domains]
        return _build_specialization_question(domains)
    else:
        return STATIC_QUESTIONS.get(q_id, STATIC_QUESTIONS["stage"])


def _get_active_questions(session: dict) -> list[str]:
    """Return the ordered list of questions that should be asked based on answers."""
    questions = [
        "stage", "job_type", "domain", "specialization", "location",
        "work_style", "company_stage", "industry", "salary",
        "role_focus", "skills", "timeline"
    ]
    
    answers = session.get("answers", {})
    
    def _to_str(val) -> str:
        if isinstance(val, list): return ", ".join(val)
        return str(val) if val else ""
        
    stage = _to_str(answers.get("stage")).lower()
    job_type = _to_str(answers.get("job_type")).lower()

    # Smart fallback: Ask foundational questions if no resume
    if not session.get("resume_uploaded", False) and not session.get("resume_raw_text"):
        questions.insert(1, "education_level")
        questions.insert(2, "years_experience")

    # Smart skipping logic
    if "student" in stage and "not graduating" in stage:
        # Internships only. Skip job_type, company_stage, salary, timeline.
        for q in ["job_type", "company_stage", "salary", "timeline", "years_experience"]:
            if q in questions: questions.remove(q)
            
    elif "intern" in job_type:
        # Skip salary and company_stage for internships
        for q in ["company_stage", "salary", "years_experience"]:
            if q in questions: questions.remove(q)
            
    elif "experienced" in stage or "3+" in stage:
        # Skip job type (assume full time)
        if "job_type" in questions: questions.remove("job_type")
        
    return questions


# ============================================================================
# Chat Endpoint — Dynamic Questions, Instant Responses
# ============================================================================

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Server-side question flow with dynamic skipping. Instant responses."""
    logger.info(f"CHAT REQUEST: session={request.session_id}, message='{request.message[:50] if request.message else ''}'")

    session = sessions.get(request.session_id)
    if not session:
        session = {
            "chat_history": [], "resume_summary": None,
            "resume_raw_text": None, "resume_uploaded": False,
            "payload": None, "answers": {},
        }
        sessions[request.session_id] = session

    if "answers" not in session:
        session["answers"] = {}

    active_questions = _get_active_questions(session)

    # ── FIRST TURN: greeting + Q1 ──
    if not request.message and not session.get("answers"):
        has_resume = bool(session.get("resume_raw_text"))
        q = _get_question(active_questions[0], session)

        if has_resume:
            preview = session.get("resume_summary", {})
            skills = preview.get("skills", [])
            skill_text = f" I can see skills like {', '.join(skills[:3])} on your resume." if skills else ""
            greeting = f"Hey! Thanks for sharing your resume.{skill_text} Let's find your perfect next role. I have a few quick questions."
        else:
            greeting = "Hey there! I'm StudojoProfiler, your career buddy. I'll ask you a few quick questions to understand what you're looking for."

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
            "total_questions": len(active_questions)
        }

    # ── RECORD USER ANSWER ──
    if request.message:
        session["chat_history"].append(ChatMessage(role="user", content=request.message))
        
        # Find the currently asked question (first one not answered)
        current_q = None
        for q_id in active_questions:
            if q_id not in session["answers"]:
                current_q = q_id
                break
                
        if current_q:
            if "," in request.message:
                session["answers"][current_q] = [a.strip() for a in request.message.split(",")]
            else:
                session["answers"][current_q] = request.message
                
        # Re-evaluate active questions since the new answer might trigger a skip
        active_questions = _get_active_questions(session)

    # ── FIND NEXT QUESTION ──
    next_q_id = None
    questions_asked_so_far = 0
    for q_id in active_questions:
        questions_asked_so_far += 1
        if q_id not in session["answers"]:
            next_q_id = q_id
            break

    # ── ALL DONE → COMPLETE ──
    if not next_q_id:
        done_msg = "Thanks for answering my questions! I have everything I need to build your career profile. Generating your report now... 📊"
        session["chat_history"].append(ChatMessage(role="assistant", content=done_msg))
        logger.info(f"COMPLETE: all {len(active_questions)} questions answered")
        return {
            "message": done_msg,
            "state": "PAYLOAD_READY",
            "mcq": None,
            "text_input": False,
            "is_complete": True,
            "questions_asked": questions_asked_so_far - 1 if questions_asked_so_far > 0 else 0,
            "total_questions": len(active_questions)
        }

    # ── SERVE NEXT QUESTION ──
    q = _get_question(next_q_id, session)
    ack = q.get("ack") or ""
    msg = f"{ack}|||{q['message']}" if ack else q["message"]

    session["chat_history"].append(ChatMessage(role="assistant", content=msg))
    logger.info(f"SERVED Q{questions_asked_so_far}/{len(active_questions)}: {next_q_id}")

    return {
        "message": msg,
        "state": "MCQ",
        "mcq": q["mcq"],
        "text_input": q["text_input"],
        "is_complete": False,
        "questions_asked": questions_asked_so_far,
        "total_questions": len(active_questions)
    }


# ============================================================================
# Payload Generation — 100% Server-Side (NO LLM)
# Maps collected answers → career ontology → roles/specializations
# Generates instantly, saves JSON + Markdown to Outputs/
# ============================================================================

class PayloadRequest(BaseModel):
    session_id: str


def _parse_salary(salary_text: str) -> dict:
    """Parse '₹4-6 LPA' into min/max ints."""
    import re
    nums = re.findall(r'[\d.]+', salary_text)
    multiplier = 1
    text_lower = salary_text.lower()
    if "lpa" in text_lower or "lac" in text_lower or "lakh" in text_lower:
        multiplier = 100000
    elif "cr" in text_lower:
        multiplier = 10000000
    if len(nums) >= 2:
        return {"min_annual_ctc": int(float(nums[0]) * multiplier), "max_annual_ctc": int(float(nums[1]) * multiplier), "currency": "INR"}
    elif len(nums) == 1:
        val = int(float(nums[0]) * multiplier)
        return {"min_annual_ctc": val, "max_annual_ctc": int(val * 1.3), "currency": "INR"}
    return {"min_annual_ctc": 0, "max_annual_ctc": 0, "currency": "INR"}


def _map_work_mode(answer: str) -> str:
    a = answer.lower()
    if "remote" in a:
        return "remote"
    elif "hybrid" in a:
        return "hybrid"
    elif "on-site" in a or "office" in a:
        return "onsite"
    return "flexible"


def _find_matching_roles(domains: list, specs: list, seniority: str) -> list:
    """Find matching roles from the ontology."""
    roles = []
    for cluster_name, specializations in CAREER_ONTOLOGY.items():
        domain_match = any(
            d.lower() in cluster_name.lower() or cluster_name.lower() in d.lower()
            for d in domains
        )
        if not domain_match:
            continue
        for spec_name, spec_roles in specializations.items():
            spec_match = any(
                s.lower() in spec_name.lower() or spec_name.lower() in s.lower()
                for s in specs
            ) if specs else True
            if spec_match:
                for role in spec_roles[:2]:
                    roles.append({
                        "title": role, "seniority": seniority,
                        "cluster": cluster_name, "specialization": spec_name,
                        "fit_score": 0.85 if spec_match else 0.65,
                        "salary_alignment": True,
                        "reasoning": f"Matches your interest in {cluster_name} / {spec_name}",
                    })
    return roles[:5]


def _generate_payload_from_answers(session: dict) -> dict:
    """Build the full CandidatePayload from collected answers + ontology."""
    import uuid as _uuid
    from datetime import datetime as _dt

    answers = session.get("answers", {})
    resume_summary = session.get("resume_summary", {}) or {}

    def _to_str(val) -> str:
        if isinstance(val, list): return ", ".join(val)
        return str(val) if val else ""

    stage = _to_str(answers.get("stage", ""))
    job_type = _to_str(answers.get("job_type", ""))
    domains = answers.get("domain", [])
    if isinstance(domains, str): domains = [domains]
    specs = answers.get("specialization", [])
    if isinstance(specs, str): specs = [specs]
    locations = answers.get("location", [])
    if isinstance(locations, str): locations = [locations]
    work_style = _to_str(answers.get("work_style", ""))
    company_stage = _to_str(answers.get("company_stage", ""))
    industries = answers.get("industry", [])
    if isinstance(industries, str): industries = [industries]
    salary_text = _to_str(answers.get("salary", ""))
    role_focus = answers.get("role_focus", [])
    if isinstance(role_focus, str): role_focus = [role_focus]
    skills = answers.get("skills", [])
    if isinstance(skills, str): skills = [skills]
    timeline = _to_str(answers.get("timeline", ""))
    education_level = _to_str(answers.get("education_level", ""))
    years_exp = _to_str(answers.get("years_experience", ""))

    seniority = "entry"
    if "intern" in job_type.lower():
        seniority = "intern"
    elif "student" in stage.lower() and "not graduating" in stage.lower():
        seniority = "intern"
    elif "experienced" in stage.lower() or "3+" in stage.lower():
        seniority = "mid"
    elif "switch" in stage.lower():
        seniority = "junior"

    matched_roles = _find_matching_roles(domains, specs, seniority)
    if not matched_roles:
        matched_roles = [
            {"title": "Business Analyst", "seniority": seniority, "cluster": "Consulting & Strategy",
             "specialization": "Management Consulting", "fit_score": 0.6, "salary_alignment": True,
             "reasoning": "General fit based on profile"},
        ]

    primary_cluster = domains[0] if domains else matched_roles[0]["cluster"]
    secondary_cluster = domains[1] if len(domains) > 1 else None

    spec_fits = []
    seen_specs = set()
    for role in matched_roles:
        if role["specialization"] not in seen_specs:
            seen_specs.add(role["specialization"])
            spec_fits.append({
                "name": role["specialization"],
                "fit_score": role["fit_score"],
                "reasoning": f"Strong match based on your interest in {role['cluster']}",
            })

    role_fits = [{"title": r["title"], "seniority": r["seniority"], "fit_score": r["fit_score"],
                  "salary_alignment": r["salary_alignment"], "reasoning": r["reasoning"]} for r in matched_roles]

    transitions = []
    if seniority in ("intern", "entry"):
        transitions.append(f"{matched_roles[0]['title']} → Senior {matched_roles[0]['title'].split()[0]} → Lead")
    if len(domains) > 1:
        transitions.append(f"Cross-domain: {domains[0]} ↔ {domains[1]}")
    transitions.append("Individual Contributor → Management track")

    risk = "medium"
    if "early" in company_stage.lower() or "seed" in company_stage.lower():
        risk = "high"
    elif "enterprise" in company_stage.lower() or "mnc" in company_stage.lower() or "government" in company_stage.lower():
        risk = "low"

    summary_parts = []
    if stage:
        summary_parts.append(stage.rstrip("."))
    if domains:
        summary_parts.append(f"interested in {', '.join(domains[:2])}")
    if locations:
        summary_parts.append(f"looking to work in {', '.join(locations[:2])}")
    profile_summary = ". ".join(summary_parts) + "." if summary_parts else "Career profile generated from conversation."

    detected_skills = resume_summary.get("skills", []) if isinstance(resume_summary, dict) else []
    if not detected_skills:
        detected_skills = skills

    return {
        "candidate_id": str(_uuid.uuid4()),
        "timestamp": _dt.now().isoformat(),
        "profile_summary": profile_summary,
        "personal_info": {
            "name": resume_summary.get("name") if isinstance(resume_summary, dict) else None,
            "email": resume_summary.get("email") if isinstance(resume_summary, dict) else None,
            "education": [{"degree": education_level, "field": "General"}] if education_level else [],
            "years_of_experience": years_exp if years_exp else None,
            "skills_detected": detected_skills[:10],
        },
        "preferences": {
            "locations": locations,
            "work_mode": _map_work_mode(work_style),
            "company_size": company_stage,
            "company_stage": company_stage,
            "industry_interests": industries,
            "salary_expectations": _parse_salary(salary_text),
            "risk_tolerance": risk,
            "timeline": timeline,
        },
        "career_analysis": {
            "primary_cluster": primary_cluster,
            "secondary_cluster": secondary_cluster,
            "specializations": spec_fits[:3],
            "recommended_roles": role_fits[:5],
            "transition_paths": transitions,
        },
        "session_metadata": {
            "resume_uploaded": session.get("resume_uploaded", False),
            "questions_answered": len(answers),
            "confidence_score": min(0.95, 0.5 + len(answers) * 0.04),
        },
    }


@app.post("/api/generate-payload")
async def generate_payload(request: PayloadRequest):
    """Generate payload server-side from collected answers (NO LLM). Instant."""
    import json
    from datetime import datetime

    session = sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if len(session.get("answers", {})) < 4:
        raise HTTPException(status_code=400, detail="Not enough answers to generate a profile")

    try:
        payload_dict = _generate_payload_from_answers(session)
        session["payload"] = payload_dict

        # ── Save to Outputs/ ──
        outputs_dir = os.path.join(os.path.dirname(__file__), "Outputs")
        os.makedirs(outputs_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        candidate_name = (payload_dict.get("personal_info", {}).get("name") or "unknown").replace(" ", "_")

        # Save raw JSON
        json_filename = f"{timestamp}_{candidate_name}.json"
        json_filepath = os.path.join(outputs_dir, json_filename)
        with open(json_filepath, "w") as jf:
            json.dump(payload_dict, jf, indent=2)

        # Save Markdown summary
        md_filename = f"{timestamp}_{candidate_name}.md"
        md_filepath = os.path.join(outputs_dir, md_filename)
        md_content = f"""---
candidate_id: {payload_dict.get('candidate_id', 'N/A')}
timestamp: {payload_dict.get('timestamp', 'N/A')}
---

# Candidate Profile: {payload_dict.get('personal_info', {}).get('name') or 'Unknown'}

## Profile Summary
{payload_dict.get('profile_summary', 'N/A')}

## Preferences
- **Locations:** {', '.join(payload_dict.get('preferences', {}).get('locations', []))}
- **Work Mode:** {payload_dict.get('preferences', {}).get('work_mode', 'N/A')}
- **Company Stage:** {payload_dict.get('preferences', {}).get('company_stage', 'N/A')}
- **Industries:** {', '.join(payload_dict.get('preferences', {}).get('industry_interests', []))}
- **Salary:** {payload_dict.get('preferences', {}).get('salary_expectations', {}).get('currency', 'INR')} {payload_dict.get('preferences', {}).get('salary_expectations', {}).get('min_annual_ctc', 0):,} - {payload_dict.get('preferences', {}).get('salary_expectations', {}).get('max_annual_ctc', 0):,}

## Recommended Roles
"""
        for role in payload_dict.get("career_analysis", {}).get("recommended_roles", []):
            md_content += f"- **{role.get('title', 'N/A')}** ({role.get('seniority', 'N/A')}) — {role.get('fit_score', 0):.0%} fit — {role.get('reasoning', '')}\n"

        md_content += f"""
## Career Transitions
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
        with open(md_filepath, "w") as f:
            f.write(md_content)

        logger.info(f"Payload saved: {json_filepath} and {md_filepath}")
        return {"payload": payload_dict, "saved_to": json_filename}

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
