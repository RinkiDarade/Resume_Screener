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

from concurrent.futures import ThreadPoolExecutor, as_completed

from typing import Optional

from pydantic import BaseModel

import time 


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
    extract_missing_skills,
    process_single_resume,
    extract_skills_from_jd
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
    save_user(user_id, full_name, email, password)  
    return RedirectResponse(url="/login", status_code=303)


#----------------------------------------------


# Login Page
@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})




@app.post("/login")
async def login(request: Request, email: str = Form(...), password: str = Form(...)):
    print("DEBUG EMAIL:", email)
    print("DEBUG PASSWORD:", password)

    user = get_user_by_email(email)
    print("DEBUG USER FROM DB:", user)

    if user and check_password_hash(user["password"], password):
        print(" Password matched!")
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


# Endpoint to process resume
@app.post("/process", response_class=HTMLResponse)
async def process_resumes(
    request: Request,
    job_description: str = Form(...),
    mandatory_keywords: str = Form(...),
    uploaded_resumes: List[UploadFile] = None
):
    MANDATORY_KEYWORDS = [kw.strip() for kw in mandatory_keywords.split(",") if kw.strip()]
    
    # Save job description
    jd_id = save_job_description(job_description, created_by=request.session.get("user", "unknown"))
    
    UPLOAD_FOLDER = "uploads"
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    # Save all files first
    file_paths = []
    for file in uploaded_resumes or []:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        file_paths.append((file_path, file.filename))

    # Process resumes in parallel
    results = []
    user_email = request.session.get("user", "unknown")
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=4) as executor:  # Adjust max_workers based on your system
        # Submit all tasks
        future_to_file = {
            executor.submit(
                process_single_resume, 
                file_path, 
                filename, 
                job_description, 
                MANDATORY_KEYWORDS, 
                jd_id, 
                user_email
            ): filename 
            for file_path, filename in file_paths
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_file):
            filename = future_to_file[future]
            try:
                result = future.result()
                if result:  # Only add successful results
                    results.append(result)
            except Exception as exc:
                print(f'Resume {filename} generated an exception: {exc}')

    # Sort by final_score
    results = sorted(results, key=lambda x: x["final_score"], reverse=True)
    

    return templates.TemplateResponse("results.html", {"request": request, "results": results})



class SkillExtractionRequest(BaseModel):
    job_description: str

@app.post("/extract-skills")
async def extract_skills_endpoint(request: SkillExtractionRequest):
    try:
        # Call your existing function
        extracted_data = extract_skills_from_jd(request.job_description)
        
        # Combine all skills from different categories
        all_skills = []
        
        # Extract skills from each field in your JSON response
        skill_fields = [
            "programming_languages", "cloud_services", "databases", 
            "devops", "big_data", "frameworks",
            "compulsory_programming_languages", "must_cloud_services",
            "must_databases", "must_devops", "must_big_data", "must_framework"
        ]
        
        for field in skill_fields:
            field_value = extracted_data.get(field, "")
            if field_value and isinstance(field_value, str):
                skills = [skill.strip() for skill in field_value.split(",") if skill.strip()]
                all_skills.extend(skills)
        
        # Remove duplicates and create comma-separated string
        unique_skills = sorted(list(set(all_skills)))
        skills_string = ", ".join(unique_skills)
        
        return {
            "success": True,
            "skills": skills_string
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "skills": ""
        }