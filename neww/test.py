from langchain_groq import ChatGroq
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import json
import re

load_dotenv()

# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────
model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,  # Lower = less hallucination
)

# ─────────────────────────────────────────────
# Global State
# ─────────────────────────────────────────────
EMAIL = URL = TOKEN = PROJECT = ""
REQ_TEXT = PERSON_TEXT = ""
TEAM_MEMBERS = {}  # { "Uvuv": "backend developer", ... }


# ─────────────────────────────────────────────
# LAYER 0 — PDF Extraction
# ─────────────────────────────────────────────
def extract_pdf_text(uploaded_file) -> str:
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += (page.extract_text() or "") + "\n"
    return text.strip()


# ─────────────────────────────────────────────
# LAYER 0B — Parse Team Members from text
# Returns { "Exact Name": "role" }
# ─────────────────────────────────────────────
def parse_team_members(person_text: str) -> dict:
    """
    Ask the LLM to extract team members as strict JSON.
    This ensures we get the exact names as they appear in the document.
    """
    prompt = f"""
You are a text parser. Extract ALL team members from the document below.
Return ONLY a valid JSON object like:
{{"Full Name": "their role", "Full Name 2": "their role"}}

Rules:
- Use the EXACT name as written in the document
- Role must be lowercase (e.g. "backend developer", "frontend developer", "tester", "database administrator")
- No explanation, no markdown, no extra text. Just JSON.

DOCUMENT:
{person_text}
"""
    res = model.invoke(prompt)
    raw = res.content.strip()
    # Strip markdown fences if present
    raw = re.sub(r"```json|```", "", raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print("⚠️ Could not parse team members JSON. Raw output:")
        print(raw)
        return {}


# ─────────────────────────────────────────────
# Send Data — call this from your UI/Streamlit
# ─────────────────────────────────────────────
def send_data(email, url, api_token, project_key, req_file, person_file):
    global EMAIL, URL, TOKEN, PROJECT
    global REQ_TEXT, PERSON_TEXT, TEAM_MEMBERS

    EMAIL = email
    URL = url
    TOKEN = api_token
    PROJECT = project_key

    # Extract text from both PDFs
    REQ_TEXT = extract_pdf_text(req_file)
    PERSON_TEXT = extract_pdf_text(person_file)

    # Parse team members into a strict dict
    TEAM_MEMBERS = parse_team_members(PERSON_TEXT)

    print(f"✅ Extracted requirements: {len(REQ_TEXT)} chars")
    print(f"✅ Extracted team members: {TEAM_MEMBERS}")

    return "done"


# ─────────────────────────────────────────────
# LAYER 1 — Build Dynamic Prompt
# ─────────────────────────────────────────────
def build_prompt(req_text: str, team_members: dict, error_feedback: str = "") -> str:
    # Build strict allowlist string
    allowlist = "\n".join(
        [f'  - "{name}" → {role}' for name, role in team_members.items()]
    )

    # Skill mapping rules
    skill_rules = """
- Assign backend / API / server tasks → backend developer
- Assign UI / frontend / screen tasks → frontend developer
- Assign database / schema / query tasks → database administrator
- Assign testing / QA / test cases tasks → tester
- If multiple people share a role → distribute evenly (round robin)
"""

    correction_block = ""
    if error_feedback:
        correction_block = f"""
⚠️ CORRECTION REQUIRED:
The previous response had these errors. Fix ONLY these issues:
{error_feedback}
"""

    prompt = f"""
You are a Jira task generator. Analyze the requirement document and generate Jira tasks.

{correction_block}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REQUIREMENT DOCUMENT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{req_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VALID ASSIGNEES (use EXACTLY these names — spelling, spacing, capitalization must match):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{allowlist}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SKILL ASSIGNMENT RULES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{skill_rules}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRICT RULES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Return ONLY valid JSON. No markdown. No explanation. No extra text.
2. Every "assignee" value MUST be one of the exact names in the VALID ASSIGNEES list above.
3. Do NOT invent, abbreviate, or modify any name.
4. Every task MUST have "status": "TODO"
5. Generate 1 Epic, multiple Stories, multiple Tasks per Story.

Return this exact structure:
{{
  "project_key": "{PROJECT if PROJECT else 'PROJ'}",
  "epic": {{
    "summary": "...",
    "description": "...",
    "epic_name": "..."
  }},
  "stories": [
    {{
      "summary": "...",
      "description": "...",
      "tasks": [
        {{
          "summary": "...",
          "description": "...",
          "assignee": "EXACT NAME FROM VALID ASSIGNEES",
          "status": "TODO"
        }}
      ]
    }}
  ]
}}
"""
    return prompt


# ─────────────────────────────────────────────
# LAYER 2 — Validate LLM Output
# Returns (is_valid: bool, errors: list[str])
# ─────────────────────────────────────────────
def validate_output(data: dict, team_members: dict) -> tuple[bool, list[str]]:
    errors = []
    valid_names = set(team_members.keys())

    stories = data.get("stories", [])
    if not stories:
        errors.append("No stories found in output.")

    for s_idx, story in enumerate(stories):
        tasks = story.get("tasks", [])
        if not tasks:
            errors.append(f"Story {s_idx+1} '{story.get('summary','')}' has no tasks.")

        for t_idx, task in enumerate(tasks):
            assignee = task.get("assignee", "")
            status = task.get("status", "")

            # Check assignee is in valid list
            if assignee not in valid_names:
                # Try to find the closest match to give helpful feedback
                errors.append(
                    f"Story {s_idx+1}, Task {t_idx+1}: "
                    f"Invalid assignee '{assignee}'. "
                    f"Must be one of: {list(valid_names)}"
                )

            # Check status
            if status != "TODO":
                errors.append(
                    f"Story {s_idx+1}, Task {t_idx+1}: "
                    f"status must be 'TODO', got '{status}'"
                )

            # Check required fields
            for field in ["summary", "description", "assignee", "status"]:
                if not task.get(field):
                    errors.append(
                        f"Story {s_idx+1}, Task {t_idx+1}: missing field '{field}'"
                    )

    return (len(errors) == 0), errors


# ─────────────────────────────────────────────
# LAYER 3 — Main Analyzer with Retry Loop
# ─────────────────────────────────────────────
def analyze_requirement_doc(max_retries: int = 3) -> dict:
    """
    Calls the LLM, validates output, and auto-corrects up to max_retries times.
    Returns the validated JSON dict.
    """
    if not REQ_TEXT or not TEAM_MEMBERS:
        raise ValueError("Call send_data() first to load documents.")

    error_feedback = ""

    for attempt in range(1, max_retries + 1):
        print(f"\n🔄 Attempt {attempt}/{max_retries}...")

        # Build prompt (with error feedback on retries)
        prompt = build_prompt(REQ_TEXT, TEAM_MEMBERS, error_feedback)

        # Call LLM
        res = model.invoke(prompt)
        raw = res.content.strip()

        # Strip markdown fences
        raw = re.sub(r"```json|```", "", raw).strip()

        # Parse JSON
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            error_feedback = f"Response was not valid JSON. Error: {e}. Return ONLY raw JSON."
            print(f"❌ JSON parse failed: {e}")
            continue

        # Validate
        is_valid, errors = validate_output(data, TEAM_MEMBERS)

        if is_valid:
            print(f"✅ Validation passed on attempt {attempt}!")
            return data
        else:
            print(f"❌ Validation failed with {len(errors)} error(s):")
            for err in errors:
                print(f"   • {err}")
            # Feed errors back into next prompt
            error_feedback = "\n".join(errors)

    raise RuntimeError(
        f"❌ Failed to generate valid output after {max_retries} attempts. "
        f"Last errors:\n{error_feedback}"
    )


# ─────────────────────────────────────────────
# Quick test — run this file directly to test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing with local PDFs...")

    result = send_data(
        email="aryansaxena1204@gmail.com",
        url="https://aryansaxena1204-1775726337759.atlassian.net",
        api_token="jira token",
        project_key="SCRUM",
        req_file="neww/requirements.pdf",
        person_file="neww/employees.pdf",
    )

    print(f"send_data: {result}")

    output = analyze_requirement_doc()

    print("\n✅ Final Output:")
    print(json.dumps(output, indent=2))