# Studojo Career Profiler

An AI-powered career profiling chatbot that collects candidate preferences via a conversational MCQ flow and generates a structured Career Intelligence Report.

## Features

- **Conversational Chat Interface** — Clean Studojo-branded UI with MCQ chips and multi-select options
- **Resume Parsing** — Upload PDF/DOCX resumes; skills, education, and contact info are auto-extracted
- **Dynamic Question Flow** — 12-question adaptive flow that skips irrelevant questions based on answers
- **Career Ontology Matching** — 500+ roles across 10 career clusters mapped via `career_ontology.py`
- **Instant Payload Generation** — No LLM calls during chat; all matching is server-side and instant
- **Exportable JSON + Markdown** — Profile reports saved to `Outputs/` folder

## Quick Start

### Prerequisites
- Python 3.11+
- pip

### Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file (optional, only needed for resume LLM features)
cp .env.example .env  # Edit with your API keys if needed

# Run the server
python main.py
```

The app will be available at **http://localhost:8000**

### Environment Variables (`.env`)

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | No | Google Gemini API key (only for advanced resume analysis) |

## Project Structure

```
├── main.py                  # FastAPI backend (chat, payload, file saving)
├── career_ontology.py       # Career cluster → specialization → role mapping
├── resume_parser.py         # PDF/DOCX text extraction + regex skill detection
├── models.py                # Pydantic data models
├── profiler_agent.py        # LLM agent (used only for advanced features)
├── image_preprocessor.py    # Image resizing utilities
├── requirements.txt         # Python dependencies
├── render.yaml              # Render.com deployment config
├── static/
│   ├── chat.js              # Frontend chat logic, MCQ rendering, progress bar
│   └── style.css            # Studojo design system CSS
├── templates/
│   └── index.html           # Main HTML template
└── Outputs/                 # Auto-generated profile reports (JSON + Markdown)
```

## How It Works

1. User uploads a resume (or skips)
2. Bot asks 10-12 MCQ questions about career preferences
3. Backend maps answers to the Career Ontology
4. Generates a Career Intelligence Report with:
   - Profile summary
   - Recommended roles with fit scores
   - Career transition paths
   - Salary alignment analysis

## Deployment

### Render.com (recommended)
The included `render.yaml` handles deployment. Push to GitHub and connect to Render.

### Docker (alternative)
```bash
docker build -t career-profiler .
docker run -p 8000:8000 career-profiler
```

## License
Proprietary — Studojo Technologies
