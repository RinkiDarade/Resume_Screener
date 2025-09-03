# main.py
import os
from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
import pandas as pd
from typing import List
from fastapi.responses import HTMLResponse, RedirectResponse
# from werkzeug.security import generate_password_hash
from werkzeug.security import check_password_hash


from resume_screener_2 import (
    get_resume_text,
    cosine_similarity_score,
    llm_score_with_reason,
    parse_resume_experience,
    extract_candidate_info_with_projects,
    build_project_skill_matrix,
    calculate_project_skill_score,
    extract_required_experience,
    extract_experience,
    classify_project_levels,
    llm_experience_verification,
    check_mandatory_keywords_in_projects,
    extract_missing_skills
)


from db import (
    save_candidate_report,
    save_job_description,
    save_candidate_status,
    get_next_candidate_id,
    get_next_jd_id,
    get_user_by_email,     
    save_user,             
    get_next_user_id       
)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

app.add_middleware(SessionMiddleware, secret_key="supersecretkey123")

app.mount("/static", StaticFiles(directory="static"), name="static")




# ---------------------------------------------------

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.post("/register")
async def register(request: Request, full_name: str = Form(...), email: str = Form(...), password: str = Form(...)):
    existing_user = get_user_by_email(email)
    if existing_user:
        return templates.TemplateResponse("register.html", {"request": request, "error": "Email already registered"})
    user_id = get_next_user_id()
    save_user(user_id, full_name, email, password)  # keep as-is
    return RedirectResponse(url="/login", status_code=303)

#----------------------------------------------


# Login Page
@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

# # Handle Login
# @app.post("/login")
# async def login(request: Request, email: str = Form(...), password: str = Form(...)):
#     # ⚡ Replace with DB lookup or actual user validation
#     if email == "admin@example.com" and password == "admin123":
#         request.session["user"] = email
#         return RedirectResponse(url="/", status_code=303)
#     return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid credentials"})


@app.post("/login")
async def login(request: Request, email: str = Form(...), password: str = Form(...)):
    print("DEBUG EMAIL:", email)
    print("DEBUG PASSWORD:", password)

    user = get_user_by_email(email)
    print("DEBUG USER FROM DB:", user)

    if user and check_password_hash(user["password"], password):
        print("✅ Password matched!")
        request.session["user"] = user["email"]
        request.session["role"] = user.get("role", "recruiter")
        return RedirectResponse(url="/", status_code=303)
    
    print("❌ Invalid credentials")
    return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid credentials"})



# Logout
@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login", status_code=303)


# Middleware helper: check if user is logged in
def require_login(request: Request):
    if "user" not in request.session:
        return False
    return True


# ---------------------------------------------------
# Homepage with JD input and file upload
# @app.get("/", response_class=HTMLResponse)
# async def home(request: Request):
#     if not require_login(request):
#         return RedirectResponse(url="/login", status_code=303)
#     return templates.TemplateResponse("index.html", {"request": request})



# Update your homepage route in main.py
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    if not require_login(request):
        return RedirectResponse(url="/login", status_code=303)
    
    # Get the logged-in user's details from database
    user_email = request.session.get("user")
    user_data = get_user_by_email(user_email) if user_email else None
    
    if not user_data:
        # If user data not found, clear session and redirect to login
        request.session.clear()
        return RedirectResponse(url="/login", status_code=303)
    
    # Pass user data to template
    return templates.TemplateResponse("index.html", {
        "request": request,
        "user_name": user_data.get("name", "User"),
        "user_email": user_data.get("email", ""),
        "user_id": user_data.get("user_id", "")
    })


# Add this route to your main.py for handling profile updates
@app.post("/update-profile")
async def update_profile(
    request: Request,
    name: str = Form(...),
    email: str = Form(...),
    company: str = Form(default="")
):
    if not require_login(request):
        return RedirectResponse(url="/login", status_code=303)
    
    current_email = request.session.get("user")
    
    try:
        # Update user in database
        from db import users_collection
        users_collection.update_one(
            {"email": current_email},
            {"$set": {
                "name": name,
                "email": email,
                "company": company,
                "updated_at": datetime.utcnow()
            }}
        )
        
        # Update session if email changed
        if email != current_email:
            request.session["user"] = email
        
        # Redirect back to dashboard with success message
        return RedirectResponse(url="/?updated=true", status_code=303)
        
    except Exception as e:
        print(f"Error updating profile: {e}")
        return RedirectResponse(url="/?error=update_failed", status_code=303)


# Endpoint to process resumes
@app.post("/process", response_class=HTMLResponse)
async def process_resumes(
    request: Request,
    job_description: str = Form(...),
    mandatory_keywords: str = Form(...),
    uploaded_resumes: List[UploadFile] = None
):
    MANDATORY_KEYWORDS = [kw.strip() for kw in mandatory_keywords.split(",") if kw.strip()]
    results = []

    
    jd_id = save_job_description(job_description, created_by=request.session.get("user", "unknown"))

    UPLOAD_FOLDER = "uploads"
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # create folder if it doesn't exist

    # Inside your /process endpoint, add this before processing each resume:
    for file in uploaded_resumes or []:
        candidate_id = get_next_candidate_id()

        # Save the uploaded file locally
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        resume_text = get_resume_text(file_path)  # Pass the local path to your resume parser
        if not resume_text:
            continue


        resume_text = resume_text.replace("\n", " ").strip()
        candidate_name, projects = extract_candidate_info_with_projects(resume_text)

        keyword_check_result = check_mandatory_keywords_in_projects(projects, MANDATORY_KEYWORDS)

        missing_skills = extract_missing_skills(keyword_check_result)
        if missing_skills:
            missing_keywords_note = ', '.join(missing_skills)
        else:
            missing_keywords_note = ""


        project_levels = classify_project_levels(projects)
        project_skill_df = build_project_skill_matrix(projects, MANDATORY_KEYWORDS)
        skill_scores, total_score = calculate_project_skill_score(project_skill_df)

        missing_keywords = [kw for kw in MANDATORY_KEYWORDS if kw.lower() not in resume_text.lower()]
        missing_note = f"Missing Keywords: {', '.join(missing_keywords)}" if missing_keywords else ""

        jd_exp = extract_required_experience(job_description)
        resume_exp_text = extract_experience(resume_text)
        resume_exp = parse_resume_experience(resume_exp_text)
        if jd_exp > 0:
            exp_warning = (
                f" Candidate has {resume_exp} years, JD requires {jd_exp} years"
                if resume_exp < jd_exp
                else f"✅ Candidate meets requirement ({resume_exp} vs {jd_exp} years)"
            )
        else:
            exp_warning = "ℹ️ JD has no explicit experience requirement"

        cos_score = cosine_similarity_score(job_description, resume_text) * 100
        genai_score, genai_reason = llm_score_with_reason(job_description, resume_text)
        authenticity_report = llm_experience_verification(resume_text)

        final_score = round((cos_score * 0.3) + (genai_score * 0.7), 2)


          # Save candidate report to DB here
        save_candidate_report(
            candidate_id=candidate_id,
            candidate_name=candidate_name,
            resume_filename=file.filename,
            experience=resume_exp,
            experience_details=resume_exp_text,
            cosine_score=round(cos_score, 2),
            genai_score=round(genai_score, 2),
            final_score=final_score,
            missing_keywords_note=", ".join(missing_skills) if missing_skills else "",
            exp_warning=exp_warning,
            projects_list=projects,  
            project_skill_df=project_skill_df.to_dict(orient="records"),   
            mandatory_skill_check_df=keyword_check_result,                
            project_level_df=project_levels,                              
            individual_skill_scores=skill_scores,                         
            genai_reason=genai_reason,
            authenticity_report=authenticity_report,
            jd_id=jd_id,
            created_by=request.session.get("user", "unknown"),
            updated_by=request.session.get("user", "unknown")
        )


        results.append({
            "filename": file.filename,
            "candidate_name": candidate_name,
            "experience": resume_exp,
            "experience_details": resume_exp_text,
            "cosine_similarity": round(cos_score, 2),
            "genai_score": round(genai_score, 2),
            "final_score": final_score,
            "missing_keywords_note": missing_note,
            "projects": projects,
            "exp_warning": exp_warning,
            "mandatory_keywords_html": keyword_check_result,
            "project_skill_html": project_skill_df.to_html(classes="table table-striped", index=True),
            "project_levels_html": pd.DataFrame(project_levels).to_html(classes="table table-striped", index=False),
            "skill_scores": skill_scores,
            "total_skill_score": total_score,
            "genai_reason": genai_reason,
            "authenticity_report": authenticity_report
        })

    # Sort by final_score
    results = sorted(results, key=lambda x: x["final_score"], reverse=True)
    return templates.TemplateResponse("results.html", {"request": request, "results": results})
