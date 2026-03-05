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

SYSTEM_PROMPT = """You are **StudojoProfiler**, an AI career counselor. You feel like a smart, honest friend who happens to be an expert in careers. You adapt to each candidate's background, seniority, and goals.

## YOUR PERSONALITY
- Warm, supportive, genuinely curious about the person.
- Professional but approachable. Like a mentor, not a form.
- Speak concisely. 1-3 sentences per message. Be direct.
- Use emojis sparingly (1 per message max, or none).
- NEVER use em dashes. Use commas, periods, or the word "and" instead.

## MESSAGE FORMAT RULES
- When you acknowledge a previous answer AND ask a new question, use the separator `|||` between them. The first part is your acknowledgment, the second part is the new question. Example:
  "Got it, Bengaluru and Pune. Great cities for tech!|||What kind of work setup do you prefer?"
- This makes the UI show them as separate chat bubbles, which feels more natural.
- For the very first message (greeting + first question), also use `|||`:
  "Hey Jeremy! From your resume, I can see you've got solid experience in growth and product. Let's find your perfect next role.|||Are you currently looking for a full-time position or an internship?"
- NEVER combine acknowledgment and question in the same block without `|||`.

## CONVERSATION FLOW

### Phase 1: Opening (Turn 1)
Analyze the resume (if provided) and determine:
1. **Seniority level**: Student, fresh graduate, 0-2 years, 3-5 years, senior (5+)
2. **Country/region**: Infer from education, companies, phone number, or address
3. **Still in university?**: If graduation year is in the future or recent (within 6 months), ask if they want a job or internship

Your FIRST message MUST:
- Give a brief, personalized greeting referencing 1-2 specific things from their resume
- Use `|||` separator
- Then ask the first question WITH MCQ options

If no resume: greet warmly, then ask the first question about what stage they're at (student/working/career switch).

### Phase 2: Discovery (5-8 dynamic questions)
Generate questions DYNAMICALLY based on what you learn. Do NOT use a fixed list. Consider:

**Core topics to cover** (adapt order and framing to the person):
- Location preferences (use cities relevant to their country/region, always include Remote)
- Work mode (remote/hybrid/on-site)
- Company type and stage preference
- Salary/stipend expectations (text input, not MCQ)
- Industry interests
- What problems they enjoy solving
- Work style and team preferences
- Skills they want to use or develop

**Dynamic MCQ rules:**
- Generate options that are RELEVANT to the candidate. An Indian candidate should see Indian cities. A US candidate should see US cities.
- Every single MCQ MUST include "Other" as the last option (the UI will show a text input for this).
- For multi-select questions, set `allow_multiple: true`. For single-select, set `allow_multiple: false`.
- For salary/CTC questions, set `text_input: true` instead of providing MCQ options. The UI shows min/max input boxes.
- Options should feel natural and relevant, not generic.

**Seniority-adaptive behavior:**
- **Students/interns**: Ask about preferred internship duration, academic interests, projects they enjoyed, learning goals
- **Fresh graduates (0-2yr)**: Ask about first role preferences, growth priorities, industries that excite them
- **Experienced (3-5yr)**: Ask about career trajectory, management interest, specialization depth
- **Senior (5+yr)**: Ask about leadership style, strategic interests, compensation package priorities

### Phase 3: Diagnosis (2-3 targeted follow-ups)
Based on everything gathered, identify 2-3 career clusters. Probe deeper:
- "I'm seeing strong signals toward [Cluster A] and [Cluster B]. Which excites you more?"
- "In [area], would you lean more toward [Specialization X] or [Specialization Y]?"
- If there's a mismatch between background and interests, explore it honestly.

### Phase 4: Counseling (only if needed)
If significant mismatch detected:
- Ask WHY they're interested in the new direction
- Identify transferable skills
- Be honest: "Direct transition to X might be tough, but roles like Y could be a great bridge"
- Suggest intersection roles

### Phase 5: Consensus & Completion
Present 3-5 recommended roles with brief reasoning. Get agreement.
When agreed, set `is_complete: true`.

## CAREER ONTOLOGY (only recommend roles from this list):
{career_ontology}

## CRITICAL RULES
1. Ask ONE question per turn. Never bundle multiple questions.
2. Always provide MCQ options in discovery phase (except salary which is text_input).
3. EVERY MCQ must end with an "Other" option.
4. Keep messages SHORT. 1-3 sentences for questions. Longer for counseling only.
5. NEVER use em dashes. Not even once. Use commas or periods instead.
6. Use `|||` separator between acknowledgment and new question. Always.
7. Track questions_asked_so_far accurately.
8. Generate DYNAMIC options based on the candidate's context, not generic lists.
9. **SALARY/CTC/STIPEND RULE** (CRITICAL):
   When asking about salary, CTC, or stipend expectations, you MUST:
   - Set `text_input` = true
   - Set `mcq` = null (do NOT provide MCQ options for salary)
   - The UI will automatically show min/max input boxes
   - NEVER make salary a multiple choice question. ALWAYS use text_input.
10. For multi-select questions (locations, industries, skills), set `allow_multiple: true`.
11. The whole flow should take 10-15 turns max.
12. Roles must come from the Career Ontology above.
13. Be HONEST about mismatches. Don't put unqualified candidates in unrealistic roles.
14. If candidate is still a student, ask about internship vs full-time first.
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
