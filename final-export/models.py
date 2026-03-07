"""
CandidateProfiler — Pydantic Models
All data structures for resume parsing, profiling, and payload generation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime
import uuid


# --- Resume Parsing Models ---

class Education(BaseModel):
    degree: str = Field(..., description="Degree name, e.g. B.Tech, MBA")
    field: str = Field(..., description="Field of study, e.g. Computer Science")
    institution: str = Field(..., description="University/college name")
    year: Optional[int] = Field(None, description="Graduation year")

class Experience(BaseModel):
    title: str = Field(..., description="Job title / role")
    company: str = Field(..., description="Company or organization name")
    duration: str = Field(..., description="Duration, e.g. '6 months', 'Jan 2023 - Jun 2023'")
    description: str = Field(..., description="Brief description of responsibilities/achievements")

class ResumeSummary(BaseModel):
    """Structured summary extracted from a resume via LLM."""
    name: Optional[str] = Field(None, description="Candidate's full name")
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    education: List[Education] = Field(default_factory=list)
    experience: List[Experience] = Field(default_factory=list)
    skills: List[str] = Field(default_factory=list, description="Technical and soft skills")
    key_strengths: List[str] = Field(default_factory=list, description="Top 3-5 standout strengths")
    career_interests: List[str] = Field(default_factory=list, description="Expressed career interests or goals")
    summary_text: str = Field(..., description="2-3 sentence overview of the candidate")


# --- Chat / Agent Models ---

class MCQOption(BaseModel):
    label: str = Field(..., description="Option label: A, B, C, D, E, etc.")
    text: str = Field(..., description="The option text")

class MCQQuestion(BaseModel):
    question: str = Field(..., description="The MCQ question text")
    options: List[MCQOption] = Field(..., description="List of answer options")
    allow_multiple: bool = Field(False, description="Whether user can select multiple options")

class AgentResponse(BaseModel):
    """What the profiling agent says/does at each turn."""
    message: str = Field(..., description="The text message to display to the candidate")
    current_state: Literal[
        "GREETING",
        "RESUME_SUMMARY",
        "MCQ",
        "DIAGNOSIS",
        "COUNSELING",
        "CONSENSUS",
        "PAYLOAD_READY"
    ] = Field(..., description="Current conversation state")
    mcq: Optional[MCQQuestion] = Field(None, description="MCQ question if current_state is MCQ")
    text_input: bool = Field(False, description="If true, show text input boxes instead of MCQ (e.g. for salary min/max)")
    is_complete: bool = Field(False, description="Whether the profiling session is complete")
    questions_asked_so_far: int = Field(0, description="Counter of questions asked")


# --- Final Payload Models ---

class SalaryRange(BaseModel):
    min_annual_ctc: int = Field(..., description="Minimum expected annual CTC")
    max_annual_ctc: int = Field(..., description="Maximum expected annual CTC")
    currency: str = Field("INR", description="Currency code")

class PersonalInfo(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    education: List[Education] = Field(default_factory=list)
    skills_detected: List[str] = Field(default_factory=list)

class CandidatePreferences(BaseModel):
    locations: List[str] = Field(..., description="Preferred job locations")
    work_mode: Literal["remote", "hybrid", "onsite", "flexible"] = Field(..., description="Work mode preference")
    company_size: str = Field(..., description="Preferred company size, e.g. '0-50', '50-200'")
    company_stage: str = Field(..., description="e.g. 'early-stage startup', 'growth-stage', 'enterprise'")
    industry_interests: List[str] = Field(..., description="Industries of interest")
    salary_expectations: SalaryRange
    risk_tolerance: Literal["high", "medium", "low"] = Field(..., description="Startup risk tolerance")
    timeline: str = Field(..., description="Job search timeline, e.g. 'immediate', '3-6 months'")

class SpecializationFit(BaseModel):
    name: str = Field(..., description="Specialization name within the career cluster")
    fit_score: float = Field(..., ge=0, le=1, description="How well the candidate fits (0-1)")
    reasoning: str = Field(..., description="Why this specialization fits")

class RoleFit(BaseModel):
    title: str = Field(..., description="Specific job role title")
    seniority: Literal["intern", "entry", "junior", "mid"] = Field("entry")
    fit_score: float = Field(..., ge=0, le=1, description="How well candidate fits (0-1)")
    salary_alignment: bool = Field(..., description="Whether role salary aligns with expectations")
    reasoning: str = Field(..., description="Why this role is recommended")

class CareerAnalysis(BaseModel):
    primary_cluster: str = Field(..., description="Primary career domain, e.g. 'Data & Analytics'")
    secondary_cluster: Optional[str] = Field(None, description="Secondary career domain")
    specializations: List[SpecializationFit] = Field(..., min_length=1)
    recommended_roles: List[RoleFit] = Field(..., min_length=1, description="3-5 specific recommended roles")
    transition_paths: List[str] = Field(default_factory=list, description="Career transition suggestions")

class SessionMetadata(BaseModel):
    resume_uploaded: bool = False
    questions_answered: int = 0
    session_duration_seconds: Optional[int] = None
    confidence_score: float = Field(0.0, ge=0, le=1)

class CandidatePayload(BaseModel):
    """The final comprehensive candidate profile payload."""
    candidate_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    profile_summary: str = Field(..., description="AI-generated 2-3 sentence candidate overview")
    personal_info: PersonalInfo
    preferences: CandidatePreferences
    career_analysis: CareerAnalysis
    session_metadata: SessionMetadata


# --- Chat History ---

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"] = Field(...)
    content: str = Field(...)
    mcq: Optional[MCQQuestion] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
