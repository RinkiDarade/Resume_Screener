import os
import tempfile
import re
import json

import streamlit as st
import PyPDF2
import docx2txt
import pandas as pd

from PyPDF2 import PdfReader

from sentence_transformers import SentenceTransformer, util
from groq import Groq
import ollama

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from dotenv import load_dotenv

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load environment variables and initialize Groq client
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# client = Groq(api_key=GROQ_API_KEY)

# Two separate Groq clients for clarity
client_extractor = Groq(api_key=GROQ_API_KEY)   # LLM-1
client_evaluator = Groq(api_key=GROQ_API_KEY)   # LLM-2

# Initialize embedding model and NLP utilities
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# # Centralized LLM call function
# def llm_call(prompt, temperature=0):
#     try:
#         response = client.chat.completions.create(
#             model="openai/gpt-oss-120b",  # Model specified here once
#             messages=[{"role": "user", "content": prompt}],
#             temperature=temperature
#         )
#         return response.choices[0].message.content.strip()
#     except Exception as e:
#         return f"LLM call error: {str(e)}"


# Text extraction functions
def extract_text_from_pdf(file):
    pdf = PyPDF2.PdfReader(file)
    return " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])


def extract_text_from_docx(file):
    return docx2txt.process(file)




def get_resume_text(uploaded_file_or_path):
    """
    Extract text from uploaded resume file (PDF or DOCX).
    Supports FastAPI UploadFile objects or local file paths (str).
    """
    text = ""

    if isinstance(uploaded_file_or_path, str):
        # Local file path
        filename = uploaded_file_or_path.lower()
        if filename.endswith(".pdf"):
            try:
                reader = PdfReader(uploaded_file_or_path)
                pages_text = [page.extract_text() for page in reader.pages if page.extract_text()]
                text = " ".join(pages_text)
            except Exception as e:
                text = f"Error reading PDF: {e}"
        elif filename.endswith(".docx"):
            try:
                text = docx2txt.process(uploaded_file_or_path)
            except Exception as e:
                text = f"Error reading DOCX: {e}"
        else:
            text = "Unsupported file format. Please upload PDF or DOCX."
    else:
        # FastAPI UploadFile
        filename = uploaded_file_or_path.filename.lower()
        if filename.endswith(".pdf"):
            try:
                uploaded_file_or_path.file.seek(0)
                reader = PdfReader(uploaded_file_or_path.file)
                pages_text = [page.extract_text() for page in reader.pages if page.extract_text()]
                text = " ".join(pages_text)
            except Exception as e:
                text = f"Error reading PDF: {e}"
        elif filename.endswith(".docx"):
            try:
                uploaded_file_or_path.file.seek(0)
                text = docx2txt.process(uploaded_file_or_path.file)
            except Exception as e:
                text = f"Error reading DOCX: {e}"
        else:
            text = "Unsupported file format. Please upload PDF or DOCX."

    return text.strip()



# Text preprocessing
def preprocess_text(text):
    text = text.lower()
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 2]
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)


# Cosine similarity calculation
def cosine_similarity_score(jd, resume):
    jd_clean = preprocess_text(jd)
    resume_clean = preprocess_text(resume)

    jd_emb = embedding_model.encode(jd_clean, convert_to_tensor=True)
    resume_emb = embedding_model.encode(resume_clean, convert_to_tensor=True)

    score = util.cos_sim(jd_emb, resume_emb)
    return float(score[0][0])


# ----------------------------
# LLM Call Functions
# ----------------------------
def llm_call_extractor(prompt, temperature=0):
    """LLM-1: Extractor / Summarizer"""
    try:
        response = client_extractor.chat.completions.create(
            model="llama-3.3-70b-versatile",   # Extractor model
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"LLM-1 call error: {str(e)}"


def llm_call_evaluator(prompt, temperature=0):
    """LLM-2: Evaluator / Recruiter"""
    try:
        response = client_evaluator.chat.completions.create(
            model="llama-3.3-70b-versatile",  # Evaluator model
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"LLM-2 call error: {str(e)}"



# def llm_call_extractor(prompt, temperature=0):
#     """LLM-1: Extractor / Summarizer using Ollama"""
#     try:
#         response = ollama.chat(
#             model="llama3.2:latest",  
#             messages=[{"role": "user", "content": prompt}],
#             options={"temperature": temperature}
#         )
#         return response["message"]["content"].strip()
#     except Exception as e:
#         return f"LLM-1 call error: {str(e)}"


# def llm_call_evaluator(prompt, temperature=0):
#     """LLM-2: Evaluator / Recruiter using Ollama"""
#     try:
#         response = ollama.chat(
#             model="llama3.2:latest",  
#             messages=[{"role": "user", "content": prompt}],
#             options={"temperature": temperature}
#         )
#         return response["message"]["content"].strip()
#     except Exception as e:
#         return f"LLM-2 call error: {str(e)}"


# wrapper
def llm_call(prompt, temperature=0, mode="extractor"):
    if mode == "extractor":
        return llm_call_extractor(prompt, temperature)
    elif mode == "evaluator":
        return llm_call_evaluator(prompt, temperature)
    else:
        return "Invalid LLM mode"


# ------------------------------


# Extract experience with llm_call
def extract_experience(resume_text):
    resume_text = resume_text.replace("\n", " ").replace("\t", " ")
    prompt = f"""
    Extract all roles, companies, and durations from the following resume text.
    also calculate total experience in years.
    Only return structured information, no extra commentary.
    Resume:
    {resume_text}
    
    Output format:
    - Total experience: <number> years
    - Roles and companies:
      - <Role 1> at <Company 1>, <Start Date> - <End Date>
      - <Role 2> at <Company 2>, <Start Date> - <End Date>
    """
    return llm_call(prompt, temperature=0, mode="extractor")


def parse_resume_experience(experience_text):
    match = re.search(r'Total experience:\s*([\d\.]+)', experience_text, flags=re.IGNORECASE)
    return float(match.group(1)) if match else 0.0


def extract_required_experience(jd_text):
    jd_text = jd_text.replace("\n", " ").lower()
    
    pattern = r'(\d+(?:\.\d+)?\+?)\s*(?:-|–|to)?\s*(\d+(?:\.\d+)?\+?)?\s*(?:years?|yrs?)'
    matches = re.findall(pattern, jd_text)

    exp_years = []
    for m in matches:
        num = m[0].replace("+", "")
        try:
            exp_years.append(float(num))
        except:
            continue

    return max(exp_years) if exp_years else 0.0


# Keyword check in projects using llm_call
def check_mandatory_keywords_in_projects(project_text, keywords):
    prompt = f"""
        You are analyzing a candidate's project section to check if the following mandatory skills are mentioned:
        {keywords}

        Instructions:
        - Consider synonyms and indirect mentions.  
        Example: "Built a web app using Flask" → Flask is considered present.  
        "Implemented LSTM" → Machine Learning is present.  
        - Output in TWO PARTS exactly:

        1. First, give two clean lists (only skills, no reasons):
        Present Skills: <comma separated list>
        Missing Skills: <comma separated list>

        2. Then, give a short reason for each missing skill (1 line each) under a section called "Reasons".

        Example Output:
        Present Skills: Python, Machine Learning
        Missing Skills: SQL, AWS

        Reasons:
        - SQL: No direct or indirect mention found.
        - AWS: No cloud-related tools mentioned.

        Project Section:
        {project_text}
        """
    return llm_call(prompt, temperature=0, mode="extractor")

# Extract "Missing Skills: <skills>" line from function output
def extract_missing_skills(text):
    for line in text.splitlines():
        line_stripped = line.strip()
        if line_stripped.lower().startswith("missing skills:"):
            # Split only once after the colon
            skills_part = line_stripped.split(":", 1)[1]
            return [skill.strip() for skill in skills_part.split(",") if skill.strip()]
    return []



# Extract projects with LLM
def extract_candidate_info_with_projects(resume_text):
    prompt = f"""
    From the resume below, extract:

    1. Candidate Name
       - If multiple names appear, pick the full name from the top of the resume or the header section.
    2. List of work-related project experiences.
       - Treat any section mentioning a "Client Name" or "Project Role" as a project.
       - For each project, provide:
         - Project Name (if not given, combine Client Name + Project Role as the name)
         - Project Description (summarize the responsibilities and technologies into a detailed description)

    Format STRICTLY as JSON like this:
    {{
      "candidate_name": "<full name of candidate>",
      "projects": [
        {{
          "name": "<project name>",
          "description": "<project description>"
        }},
        ...
      ]
    }}

    Resume:
    {resume_text}
    """
    
    response = llm_call(prompt, temperature=0, mode="extractor")
    
    

    try:
        # Clean response (remove markdown code fences if present)
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
        response = response.strip()

        # Parse JSON
        data = json.loads(response)

        candidate_name = data.get("candidate_name", "").strip()
        projects = [p for p in data.get("projects", []) if p.get("name") and p.get("description")]
        return candidate_name, projects

    except Exception as e:
        print("Error during extraction:", e)
        print("Raw LLM Response:", response)  # debugging
        return "", []


# Build project skill matrix
def build_project_skill_matrix(projects, mandatory_skills):
    matrix = []
    for project in projects:
        skill_status = {skill.lower(): 0 for skill in mandatory_skills}
        response_text = check_mandatory_keywords_in_projects(project['description'], mandatory_skills)
        present_line = [line for line in response_text.splitlines() if line.startswith("Present Skills:")]
        if present_line:
            skills_str = present_line[0].split("Present Skills:")[1]
            present_skills = [s.strip().lower() for s in skills_str.split(",") if s.strip()]
            for skill in present_skills:
                if skill in skill_status:
                    skill_status[skill] = 1
        matrix.append(skill_status)

    project_names = [p['name'] for p in projects]
    df = pd.DataFrame(matrix)
    df.index = project_names
    return df


# Calculate project skill scores
def calculate_project_skill_score(skill_project_df):
    total_projects = skill_project_df.shape[0]
    total_skills = skill_project_df.shape[1]

    if total_skills == 0 or total_projects == 0:
        return {}, 0.0

    base_weight = 100 / total_skills
    skill_scores = {}

    for skill in skill_project_df.columns:
        present_count = skill_project_df[skill].sum()
        factor = present_count / total_projects
        score = round(base_weight * factor, 2)
        skill_scores[skill] = {"count": int(present_count), "score": score}

    total_score = round(sum(v['score'] for v in skill_scores.values()), 2)
    return skill_scores, total_score


# Additional LLM functions unchanged but call `llm_call`:
def classify_project_levels(projects):
    prompt = (
        "You are a career coach reviewing a candidate's resume. "
        "For each project listed below, classify the skill level required to complete it as "
        "**Beginner**, **Intermediate**, or **Advanced**.\n\n"
        "Also, provide a **very short reason (max 1 short sentence)** focusing only on key skills or methods involved.\n"
        "Avoid long explanations or descriptions.\n\n"
        "Respond in the following JSON format:\n"
        '[{"name": "<Project Name>", "level": "<Beginner | Intermediate | Advanced>", "reason": "<Short explanation>"}]\n\n'
    )
    for project in projects:
        prompt += f"Title: {project['name']}\nDescription: {project['description']}\n\n"

    response = llm_call(prompt,temperature=0, mode="evaluator")
    try:
        return json.loads(response)
    except Exception as e:
        print("Error parsing LLM response:", e)
        return []


def llm_experience_verification(resume_text):
    prompt = f"""
        You are an AI resume authenticity evaluator. Be terse.

        Evaluate the resume against these 6 checks:
        1) Unrealistic timelines → Someone claiming 7 years of ML experience but graduated 2 years ago.
        2) Repeated job titles → Copy-paste of same role description for different companies
        3) Chronology check → Are there logical gaps in employment history (e.g., “2019–2023” in one job and “2021–2022” in another overlapping role)?
        4) Unaccredited institutes → Training or degree from unrecognized institutes.
        5) Misalignment → Example: claiming expertise in “10 years of cloud experience” when AWS started in 2006 (impossible for younger candidates).
        6) Too many “senior” titles in a short career.
        7) Missing details (company size, project scope).

        Special (REPORT ONLY, DO NOT affect score):
        - Inconsistent personal details. Do not mention any email addresses.
        - Exclude cloud platform certifications from scoring.

        Scoring:
        - RISK SCORE 0–100.
        - Low 0–30 (0–1 minor issues), Medium 31–60 (2–3 issues), High 61–100 (4+ or severe).

        OUTPUT RULES (strict):
        - Respond ONLY in this format:
        RISK SCORE: <number>
        FINDINGS:
        - <one-sentence finding>
        - <one-sentence finding>
        ...
        - Max 6 findings. One sentence each (≤15 words). No extras, no advice, no sections beyond these two lines.
        - If no issues: include one finding, “No major issues detected.”

        Resume:
        {resume_text}
    """
    return llm_call(prompt, temperature=0,mode="evaluator")


def llm_score_with_reason(jd, resume):
    prompt = f"""
    You are an AI recruiter. Compare the following job description and resume.

    1. Provide a match score between 0 and 100.
    2. Summarize the comparison in short, **bullet points** only:
       - Key strengths (matching skills)
       - Missing/weak skills
    3. Keep explanation short and valuable (max 4-5 bullets).
    4. Format your response exactly as:
       SCORE: <number>
       REASON:
       - <point 1>
       - <point 2>
       - <point 3>

    Job Description:
    {jd}

    Resume:
    {resume}
    """
    response = llm_call(prompt, temperature=0, mode="evaluator")
    lines = response.split("\n")
    score_line = [l for l in lines if "SCORE" in l.upper()]
    reason_lines = [l for l in lines if l.strip().startswith("-")]

    score = float("".join(filter(lambda c: c.isdigit() or c == ".", score_line[0]))) if score_line else 0.0
    reason = "\n".join(reason_lines) if reason_lines else "No explanation provided."
    return score, reason
