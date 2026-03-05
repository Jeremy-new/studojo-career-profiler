"""
CandidateProfiler — Profiling Agent (Dynamic Mode)
Uses Azure OpenAI (via instructor) with a dynamic, personalized system prompt.
All questions are LLM-generated based on resume context. No hardcoded MCQs.
"""

import os
import logging
import instructor
from openai import AzureOpenAI
from dotenv import load_dotenv
from models import AgentResponse, ChatMessage

load_dotenv()
logger = logging.getLogger(__name__)

# ============================================================================
# Azure OpenAI Client (instructor-powered)
# ============================================================================

_client = None


def _get_client():
    global _client
    if _client is None:
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")

        if not endpoint or not api_key:
            raise ValueError("AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must be set")

        base_client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )
        _client = instructor.from_openai(base_client)
    return _client


def _get_model():
    return os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini")


# ============================================================================
# Career Ontology
# ============================================================================

CAREER_ONTOLOGY_FILE = os.path.join(os.path.dirname(__file__), "career_ontology.json")


def get_ontology_as_text() -> str:
    import json
    try:
        with open(CAREER_ONTOLOGY_FILE, "r") as f:
            ontology = json.load(f)
        lines = []
        for cluster in ontology.get("career_clusters", []):
            lines.append(f"\n### {cluster['cluster_name']}")
            for role in cluster.get("roles", []):
                seniority = ", ".join(role.get("seniority_levels", []))
                lines.append(f"- {role['title']} ({seniority})")
        return "\n".join(lines)
    except Exception as e:
        logger.warning(f"Could not load career ontology: {e}")
        return "(Career ontology not available)"


# ============================================================================
# SYSTEM PROMPT: Dynamic, Personalized Career Counselor
# ============================================================================

SYSTEM_PROMPT = """You are **StudojoProfiler**, a career profiling chatbot. Your ONLY job is to understand the candidate's background, preferences, and career goals in 8 questions, then end the conversation.

## YOUR MISSION
Collect enough information to generate a structured career profile payload. You are NOT a career counselor, mentor, or advisor. Do NOT give tips, templates, comparisons, career advice, or do anything beyond asking profiling questions.

## YOUR PERSONALITY
- Warm and concise. 1 short sentence to acknowledge, then the next question.
- NEVER use em dashes. Use commas or periods instead.
- Use emojis sparingly (max 1 per message, or none).

## MESSAGE FORMAT
- Use `|||` to separate acknowledgment from the next question. Example:
  "Great, Bengaluru it is.|||What type of company do you prefer?"
- NEVER combine both in one block without `|||`.

## QUESTION PLAN (exactly 8 questions, adapt wording to context)

### Q1: Stage (hardcoded by system, skip if already answered)
"Which of these best describes you right now?" (student/grad/experienced/switching)

### Q2: Job type (if student/grad)
"Are you looking for an internship or full-time role?" OR skip if experienced.

### Q3: Location preferences
"Which cities/regions would you prefer to work in?" (multi-select, include Remote)

### Q4: Company stage and size
"What type of company do you want to join?" (startup/growth/enterprise, multi-select)

### Q5: Industry interests
"Which industries excite you most?" (multi-select, 5-7 options based on resume/context)

### Q6: Salary/CTC expectations
"What's your expected salary range?" (text_input: true, mcq: null)

### Q7: Role focus / what they enjoy
"What kind of work do you enjoy most?" (building product, analyzing data, managing people, etc.)

### Q8: Skills to use or grow
"Which skills do you want to use or develop?" (multi-select based on resume + context)

After Q8: Set `is_complete: true`. Do NOT ask more questions. Say something like:
"Thanks! I have everything I need to build your career profile. Generating your report now..."

## DYNAMIC MCQ RULES
- Generate options RELEVANT to the candidate (Indian cities for Indian candidates, etc.)
- Every MCQ MUST end with "Other" as the last option.
- Multi-select questions: set `allow_multiple: true`.
- Salary/CTC: ALWAYS use `text_input: true`, `mcq: null`.

## CAREER ONTOLOGY (reference for recommended roles):
{career_ontology}

## FORBIDDEN (NEVER DO THESE)
- Do NOT give career advice, tips, or guidance.
- Do NOT compare roles, explain day-to-day tasks, or create templates.
- Do NOT go beyond 8 questions. After Q8, you MUST set is_complete: true.
- Do NOT ask clarifying follow-ups like "which excites you more?" or "do you prefer X or Y?" — these count as extra questions.
- Do NOT spiral into sub-questions. Each question should cover ONE topic and move on.
- Do NOT promise to "prepare" or "provide" anything. You are collecting data, not delivering insight.
- Do NOT repeat yourself or paraphrase the user's answer back at length.

## CRITICAL RULES (VIOLATION BREAKS THE UI)
1. Ask ONE question per turn. Never bundle multiple questions.
2. EVERY response MUST include mcq options (except salary = text_input). NO EXCEPTIONS.
3. EVERY MCQ must end with "Other".
4. Keep acknowledgments to ONE short sentence. No paragraphs.
5. Use `|||` separator between acknowledgment and new question. Always.
6. Track questions_asked_so_far accurately (increment by 1 each turn).
7. After question 8 (questions_asked_so_far >= 8), set `is_complete: true` immediately.
8. NEVER return a response without mcq/text_input unless is_complete is true.
9. current_state should be "MCQ" for questions 1-8, "PAYLOAD_READY" when complete.
10. The ENTIRE conversation must be 8 questions. Not 9, not 15, not 30. Exactly 8.
"""


# ============================================================================
# Message Builder
# ============================================================================

def build_messages(
    chat_history: list[ChatMessage],
    resume_summary: dict | None = None,
    resume_raw_text: str | None = None,
) -> list[dict]:
    """Build the message list for the LLM call."""
    system_content = SYSTEM_PROMPT.format(career_ontology=get_ontology_as_text())

    # Add resume context (raw text, since we skip LLM summarization)
    if resume_raw_text:
        truncated = resume_raw_text[:4000]  # Keep more context for better understanding
        system_content += f"\n\n## CANDIDATE'S RESUME (raw text):\n{truncated}\n"

        # Add quick-extracted metadata if available
        if resume_summary and isinstance(resume_summary, dict):
            if resume_summary.get("name"):
                system_content += f"\nDetected name: {resume_summary['name']}"
            if resume_summary.get("email"):
                system_content += f"\nDetected email: {resume_summary['email']}"
            if resume_summary.get("skills"):
                system_content += f"\nDetected skills: {', '.join(resume_summary['skills'])}"

    messages = [
        {"role": "system", "content": system_content},
    ]

    if not chat_history:
        messages.append({"role": "user", "content": "Start the profiling session."})
    else:
        for msg in chat_history:
            messages.append({"role": msg.role, "content": msg.content})

    return messages


# ============================================================================
# Agent Response
# ============================================================================

def get_agent_response(
    chat_history: list[ChatMessage],
    resume_summary: dict | None = None,
    resume_raw_text: str | None = None,
) -> AgentResponse:
    """
    Get the next agent response. Uses instructor for structured output.
    """
    client = _get_client()
    model = _get_model()
    messages = build_messages(chat_history, resume_summary, resume_raw_text)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            response_model=AgentResponse,
            max_completion_tokens=1500,
        )
        logger.info(f"Agent response: state={response.current_state}, "
                    f"has_mcq={response.mcq is not None}, "
                    f"complete={response.is_complete}")
        return response

    except Exception as e:
        logger.error(f"Agent error: {e}")
        return AgentResponse(
            message="I hit a snag processing that. Could you try again?",
            current_state="MCQ",
            questions_asked_so_far=len([m for m in chat_history if m.role == "assistant"]),
        )


# ============================================================================
# Final Payload Generation
# ============================================================================

PAYLOAD_PROMPT = """You are a career analysis AI. Read the entire conversation between the career counselor and the candidate below. Extract ALL relevant information and generate a comprehensive CandidatePayload.

## INSTRUCTIONS
- Fill in ALL fields based on what was discussed in the conversation.
- For salary, use the numbers mentioned or estimate based on the role and seniority.
- Recommend 3-5 roles with fit scores (0-1) and reasoning.
- Be specific in your analysis and recommendations.
- If information was not discussed (e.g. name, email), leave those fields as null/empty.
- For specializations, identify 2-3 based on the candidate's skills and interests.
- For transition_paths, suggest 2-3 career progression paths.
- profile_summary should be a concise 2-3 sentence overview.
"""


def generate_final_payload(
    chat_history: list[ChatMessage],
    resume_summary: dict | None = None,
    resume_raw_text: str | None = None,
    resume_uploaded: bool = False,
) -> "CandidatePayload":
    """
    Generate the final candidate profile payload from the conversation.
    Uses instructor for structured output with the CandidatePayload model.
    """
    from models import CandidatePayload

    client = _get_client()
    model = _get_model()

    # Build the conversation transcript
    transcript = ""
    for msg in chat_history:
        role_label = "Counselor" if msg.role == "assistant" else "Candidate"
        transcript += f"\n{role_label}: {msg.content}\n"

    # Add resume context
    resume_context = ""
    if resume_raw_text:
        resume_context = f"\n\n## CANDIDATE'S RESUME:\n{resume_raw_text[:4000]}\n"
    if resume_summary and isinstance(resume_summary, dict):
        if resume_summary.get("name"):
            resume_context += f"\nName: {resume_summary['name']}"
        if resume_summary.get("email"):
            resume_context += f"\nEmail: {resume_summary['email']}"
        if resume_summary.get("skills"):
            resume_context += f"\nSkills: {', '.join(resume_summary['skills'])}"

    messages = [
        {"role": "system", "content": PAYLOAD_PROMPT + resume_context},
        {"role": "user", "content": f"Here is the full conversation transcript:\n{transcript}\n\nGenerate the CandidatePayload based on this conversation."},
    ]

    try:
        payload = client.chat.completions.create(
            model=model,
            messages=messages,
            response_model=CandidatePayload,
            max_completion_tokens=1500,
        )
        logger.info(f"Payload generated: {payload.candidate_id}")
        return payload
    except Exception as e:
        logger.error(f"Payload generation error: {e}")
        raise

