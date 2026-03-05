"""
CandidateProfiler — Profiling Agent (Direct JSON Mode)
Uses Azure OpenAI with JSON response format (NOT instructor/tool-calling).
This is dramatically faster with reasoning models like gpt-5-mini.
"""

import os
import json
import logging
from openai import AzureOpenAI
from dotenv import load_dotenv
from models import AgentResponse, MCQQuestion, MCQOption, ChatMessage

load_dotenv()
logger = logging.getLogger(__name__)

# ============================================================================
# Azure OpenAI Client (direct, no instructor)
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

        _client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )
    return _client


def _get_model():
    return os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini")


# ============================================================================
# Career Ontology
# ============================================================================


def get_ontology_as_text() -> str:
    try:
        from career_ontology import CAREER_ONTOLOGY
        lines = []
        for cluster_name, specializations in CAREER_ONTOLOGY.items():
            lines.append(f"\n### {cluster_name}")
            for spec_name, roles in specializations.items():
                lines.append(f"  {spec_name}: {', '.join(roles[:3])}")
        return "\n".join(lines)
    except Exception as e:
        logger.warning(f"Could not load career ontology: {e}")
        return "(Career ontology not available)"


# ============================================================================
# SYSTEM PROMPT
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
- Do NOT ask clarifying follow-ups.
- Do NOT spiral into sub-questions.
- Do NOT promise to "prepare" or "provide" anything.
- Do NOT repeat yourself or paraphrase the user's answer back at length.

## RESPONSE FORMAT
You MUST respond with a JSON object matching this exact schema:
{{
  "message": "Your text message here (use ||| to separate acknowledgment from question)",
  "current_state": "MCQ" or "PAYLOAD_READY",
  "mcq": {{
    "question": "The question text",
    "options": [
      {{"label": "A", "text": "Option text"}},
      {{"label": "B", "text": "Option text"}},
      {{"label": "C", "text": "Other"}}
    ],
    "allow_multiple": false
  }},
  "text_input": false,
  "is_complete": false,
  "questions_asked_so_far": 2
}}

For salary questions, set mcq to null and text_input to true.
When is_complete is true, set mcq to null and text_input to false.
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
        truncated = resume_raw_text[:4000]
        system_content += f"\n\n## CANDIDATE'S RESUME (raw text):\n{truncated}\n"

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
# Parse JSON Response into AgentResponse
# ============================================================================

def _parse_llm_json(raw_text: str, chat_history: list[ChatMessage]) -> AgentResponse:
    """Parse the LLM's JSON string into an AgentResponse model."""
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown code blocks
        import re
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw_text, re.DOTALL)
        if match:
            data = json.loads(match.group(1))
        else:
            logger.error(f"Could not parse LLM response as JSON: {raw_text[:300]}")
            return AgentResponse(
                message="Could you repeat that? I had trouble processing.",
                current_state="MCQ",
                questions_asked_so_far=len([m for m in chat_history if m.role == "assistant"]),
            )

    # Build MCQ if present
    mcq = None
    if data.get("mcq"):
        mcq_data = data["mcq"]
        options = [MCQOption(label=o.get("label", chr(65 + i)), text=o["text"])
                   for i, o in enumerate(mcq_data.get("options", []))]
        mcq = MCQQuestion(
            question=mcq_data.get("question", ""),
            options=options,
            allow_multiple=mcq_data.get("allow_multiple", False),
        )

    return AgentResponse(
        message=data.get("message", ""),
        current_state=data.get("current_state", "MCQ"),
        mcq=mcq,
        text_input=data.get("text_input", False),
        is_complete=data.get("is_complete", False),
        questions_asked_so_far=data.get("questions_asked_so_far", 0),
    )


# ============================================================================
# Agent Response (Direct JSON mode - NO instructor)
# ============================================================================

def get_agent_response(
    chat_history: list[ChatMessage],
    resume_summary: dict | None = None,
    resume_raw_text: str | None = None,
) -> AgentResponse:
    """
    Get the next agent response using direct JSON mode.
    Much faster than instructor's tool-calling for reasoning models.
    """
    client = _get_client()
    model = _get_model()
    messages = build_messages(chat_history, resume_summary, resume_raw_text)

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            max_completion_tokens=16000,
        )
        raw = completion.choices[0].message.content
        logger.info(f"LLM raw response length: {len(raw) if raw else 0} chars")

        if not raw:
            logger.error("LLM returned empty response")
            return AgentResponse(
                message="Could you repeat that? I had trouble processing.",
                current_state="MCQ",
                questions_asked_so_far=len([m for m in chat_history if m.role == "assistant"]),
            )

        response = _parse_llm_json(raw, chat_history)
        logger.info(f"Agent response: state={response.current_state}, "
                    f"has_mcq={response.mcq is not None}, "
                    f"complete={response.is_complete}")
        return response

    except Exception as e:
        logger.error(f"Agent error: {e}")
        return AgentResponse(
            message="I hit a snag processing that. Cou