# main.py
import os
from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
import pandas as pd
import time
from typing import List
from datetime import datetime
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, RedirectResponse
# from werkzeug.security import generate_password_hash
from werkzeug.security import check_password_hash


from concurrent.futures import ThreadPoolExecutor, as_completed


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

# @app.post("/process", response_class=HTMLResponse)
# async def process_resumes(
#     request: Request,
#     job_description: str = Form(...),
#     mandatory_keywords: str = Form(...),
#     uploaded_resumes: List[UploadFile] = None
# ):
#     start_time = time.time()  # ⏱️ Start timer

#     MANDATORY_KEYWORDS = [kw.strip() for kw in mandatory_keywords.split(",") if kw.strip()]
#     results = []

#     jd_id = save_job_description(job_description, created_by=request.session.get("user", "unknown"))

#     UPLOAD_FOLDER = "uploads"
#     os.makedirs(UPLOAD_FOLDER, exist_ok=True)

#     for file in uploaded_resumes or []:
#         candidate_id = get_next_candidate_id()

#         file_path = os.path.join(UPLOAD_FOLDER, file.filename)
#         with open(file_path, "wb") as f:
#             f.write(await file.read())

#         resume_text = get_resume_text(file_path)
#         if not resume_text:
#             continue

#         resume_text = resume_text.replace("\n", " ").strip()
#         candidate_name, projects = extract_candidate_info_with_projects(resume_text)

#         keyword_check_result = check_mandatory_keywords_in_projects(projects, MANDATORY_KEYWORDS)
#         missing_skills = extract_missing_skills(keyword_check_result)
#         missing_keywords_note = ', '.join(missing_skills) if missing_skills else ""

#         project_levels = classify_project_levels(projects)
#         project_skill_df = build_project_skill_matrix(projects, MANDATORY_KEYWORDS)
#         skill_scores, total_score = calculate_project_skill_score(project_skill_df)

#         missing_keywords = [kw for kw in MANDATORY_KEYWORDS if kw.lower() not in resume_text.lower()]
#         missing_note = f"Missing Keywords: {', '.join(missing_keywords)}" if missing_keywords else ""

#         jd_exp = extract_required_experience(job_description)
#         resume_exp_text = extract_experience(resume_text)
#         resume_exp = parse_resume_experience(resume_exp_text)
#         if jd_exp > 0:
#             exp_warning = (
#                 f" Candidate has {resume_exp} years, JD requires {jd_exp} years"
#                 if resume_exp < jd_exp
#                 else f"✅ Candidate meets requirement ({resume_exp} vs {jd_exp} years)"
#             )
#         else:
#             exp_warning = "ℹ️ JD has no explicit experience requirement"

#         cos_score = cosine_similarity_score(job_description, resume_text) * 100
#         genai_score, genai_reason = llm_score_with_reason(job_description, resume_text)
#         authenticity_report = llm_experience_verification(resume_text)

#         final_score = round((cos_score * 0.3) + (genai_score * 0.7), 2)

#         save_candidate_report(
#             candidate_id=candidate_id,
#             candidate_name=candidate_name,
#             resume_filename=file.filename,
#             experience=resume_exp,
#             experience_details=resume_exp_text,
#             cosine_score=round(cos_score, 2),
#             genai_score=round(genai_score, 2),
#             final_score=final_score,
#             missing_keywords_note=", ".join(missing_skills) if missing_skills else "",
#             exp_warning=exp_warning,
#             projects_list=projects,
#             project_skill_df=project_skill_df.to_dict(orient="records"),
#             mandatory_skill_check_df=keyword_check_result,
#             project_level_df=project_levels,
#             individual_skill_scores=skill_scores,
#             genai_reason=genai_reason,
#             authenticity_report=authenticity_report,
#             jd_id=jd_id,
#             created_by=request.session.get("user", "unknown"),
#             updated_by=request.session.get("user", "unknown")
#         )

#         results.append({
#             "filename": file.filename,
#             "candidate_name": candidate_name,
#             "experience": resume_exp,
#             "experience_details": resume_exp_text,
#             "cosine_similarity": round(cos_score, 2),
#             "genai_score": round(genai_score, 2),
#             "final_score": final_score,
#             "missing_keywords_note": missing_note,
#             "projects": projects,
#             "exp_warning": exp_warning,
#             "mandatory_keywords_html": keyword_check_result,
#             "project_skill_html": project_skill_df.to_html(classes="table table-striped", index=True),
#             "project_levels_html": pd.DataFrame(project_levels).to_html(classes="table table-striped", index=False),
#             "skill_scores": skill_scores,
#             "total_skill_score": total_score,
#             "genai_reason": genai_reason,
#             "authenticity_report": authenticity_report
#         })

#     results = sorted(results, key=lambda x: x["final_score"], reverse=True)

#     total_time = time.time() - start_time  # ⏱️ End timer
#     print(f"⚡ Total processing time: {total_time:.2f} seconds for {len(uploaded_resumes or [])} resumes")  

#     return templates.TemplateResponse("results.html", {"request": request, "results": results})














#===============================================================================================================================================================================





# @app.post("/process", response_class=HTMLResponse)
# async def process_resumes(
#     request: Request,
#     job_description: str = Form(...),
#     mandatory_keywords: str = Form(...),
#     uploaded_resumes: List[UploadFile] = None
# ):
#     start_time = time.time()  #  Start timer

#     # ✅ Extract skills from JD
#     parsed_skills = extract_skills_from_jd(job_description)
#     print("parsed_skills:", parsed_skills)

#     # Collect mandatory keywords (you can tweak logic here)
#     MANDATORY_KEYWORDS = []
#     if parsed_skills.get("compulsory_programming_languages"):
#         MANDATORY_KEYWORDS.extend(parsed_skills["compulsory_programming_languages"].split(","))
#     if parsed_skills.get("must_cloud_services"):
#         MANDATORY_KEYWORDS.extend(parsed_skills["must_cloud_services"].split(","))
#     if parsed_skills.get("must_databases"):
#         MANDATORY_KEYWORDS.extend(parsed_skills["must_databases"].split(","))
#     if parsed_skills.get("must_devops"):
#         MANDATORY_KEYWORDS.extend(parsed_skills["must_devops"].split(","))
#     if parsed_skills.get("must_big_data"):
#         MANDATORY_KEYWORDS.extend(parsed_skills["must_big_data"].split(","))
#     if parsed_skills.get("must_framework"):
#         MANDATORY_KEYWORDS.extend(parsed_skills["must_framework"].split(","))

#     # Strip and clean keywords
#     MANDATORY_KEYWORDS = [kw.strip() for kw in MANDATORY_KEYWORDS if kw.strip()]

#     # Save job description
#     jd_id = save_job_description(job_description, created_by=request.session.get("user", "unknown"))
    

#     #mera function elastic search vala  call hoga, parsed skills ka array as input to my function and it will return list of jd 


#     UPLOAD_FOLDER = "uploads"
#     os.makedirs(UPLOAD_FOLDER, exist_ok=True)

#     # Save all files first
#     file_paths = []
#     for file in uploaded_resumes or []:
#         candidate_id = get_next_candidate_id()
#         file_path = os.path.join(UPLOAD_FOLDER, file.filename)
#         with open(file_path, "wb") as f:
#             f.write(await file.read())
#         file_paths.append((file_path, file.filename))

#     # Process resumes in parallel
#     results = []
#     user_email = request.session.get("user", "unknown")
    
#     with ThreadPoolExecutor(max_workers=4) as executor:  
#         future_to_file = {
#             executor.submit(
#                 process_single_resume, 
#                 file_path, 
#                 filename, 
#                 job_description, 
#                 MANDATORY_KEYWORDS, 
#                 jd_id, 
#                 user_email
#             ): filename 
#             for file_path, filename in file_paths
#         }
        
#         for future in as_completed(future_to_file):
#             filename = future_to_file[future]
#             try:
#                 result = future.result()
#                 if result:
#                     results.append(result)
#             except Exception as exc:
#                 print(f'❌ Resume {filename} generated an exception: {exc}')

#     # Sort by final_score
#     results = sorted(results, key=lambda x: x["final_score"], reverse=True)
    
#     # Clean up uploaded files
#     for file_path, _ in file_paths:
#         try:
#             os.remove(file_path)
#         except:
#             pass

#     total_time = time.time() - start_time  # ⏱️ End timer
#     print(f"⚡ Processed {len(file_paths)} resumes in {total_time:.2f} seconds using ThreadPoolExecutor")

#     return templates.TemplateResponse("results.html", {"request": request, "results": results})







# BELOW WORKING





from fastapi import Request, Form, UploadFile
from fastapi.responses import HTMLResponse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os, time, tempfile, json
from typing import List
from utils.elastic_search_inbuild import ElasticSearchHandler


# @app.post("/process", response_class=HTMLResponse)
# async def process_resumes(
#     request: Request,
#     job_description: str = Form(...),
#     mandatory_keywords: str = Form(""),
#     # uploaded_resumes: List[UploadFile] = None,
# ):
#     start_time = time.time()

#     parsed_skills = extract_skills_from_jd(job_description)
#     print("parsed_skills:", parsed_skills)

#     # Collect keywords
#     mandatory_keywords_list = [kw.strip() for kw in mandatory_keywords.split(",") if kw.strip()]
#     MANDATORY_KEYWORDS = []
#     for key in [
#         "compulsory_programming_languages",
#         "must_cloud_services",
#         "must_databases",
#         "must_devops",
#         "must_big_data",
#         "must_framework",
#     ]:
#         if parsed_skills.get(key):
#             MANDATORY_KEYWORDS.extend(parsed_skills[key].split(","))
#     MANDATORY_KEYWORDS = [kw.strip() for kw in MANDATORY_KEYWORDS if kw.strip()]
#     MANDATORY_KEYWORDS = list(set(MANDATORY_KEYWORDS + mandatory_keywords_list))

#     # Save JD
#     try:
#         jd_id = save_job_description(job_description, created_by=request.session.get("user", "unknown"))
#     except Exception as e:
#         print(f"❌ Failed to save job description: {e}")
#         jd_id = None

#     # ✅ Call Elasticsearch
#     try:
#         es_handler = ElasticSearchHandler(
#             hostname="localhost",
#             port=9200,
#             scheme="http",
#             index_name="parsed_resumes_textfield"
#         )
#         resumes_from_es = es_handler.search_resumes(
#             min_exp_range=parsed_skills.get("min_exp_range", 0) or 0,
#             max_exp_range=parsed_skills.get("max_exp_range", 50) or 50,
#             compulsory_programming_languages=parsed_skills.get("compulsory_programming_languages", ""),
#             programming_languages_compulsory=parsed_skills.get("programming_languages_compulsory", False),
#             programming_languages=parsed_skills.get("programming_languages", ""),
#             cloud_services_compulsory=parsed_skills.get("cloud_services_compulsory", False),
#             must_cloud_services=parsed_skills.get("must_cloud_services", ""),
#             cloud_services=parsed_skills.get("cloud_services", ""),
#             databases_compulsory=parsed_skills.get("databases_compulsory", False),
#             must_databases=parsed_skills.get("must_databases", ""),
#             databases=parsed_skills.get("databases", ""),
#             devops_compulsory=parsed_skills.get("devops_compulsory", False),
#             must_devops=parsed_skills.get("must_devops", ""),
#             devops=parsed_skills.get("devops", ""),
#             big_data_compulsory=parsed_skills.get("big_data_compulsory", False),
#             must_big_data=parsed_skills.get("must_big_data", ""),
#             big_data=parsed_skills.get("big_data", ""),
#             framework_compulsory=parsed_skills.get("frameworks_compulsory", False),
#             must_framework=parsed_skills.get("must_framework", ""),
#             framework=parsed_skills.get("frameworks", "")
#         )
#         print("📂 Resumes from ES:", resumes_from_es)
#         resume_path=[]
#         resume_path = [details['resume_path'] for details in resumes_from_es.values()]
#         print("---------------------------------------------------")
#         print("file_path:", resume_path)
#         print("---------------------------------")
            
#         # print("📂 Final Resume Paths (to use further):", resume_paths)
#     except Exception as e:
#         print(f"❌ Error fetching resumes from ES: {e}")
#         resumes_from_es = []

#     #comma seperated file path and pass to upload resumes in list

#     # Temp save uploaded resumes
#     temp_dir = tempfile.mkdtemp(prefix="resumes_")
#     file_paths = []

#     for file in resume_path or []:
#         candidate_id = get_next_candidate_id()
#         # file_names = [os.path.basename(path) for path in resume_path]
#         # print("file_names:",file_names)
#         file_path = file
#         file_name = os.path.basename(file_path)
#         print("++++++++++++++++++++++++++++++++++++++++++")
#         print("file path:", file_name)

#         # with open(file_path, "wb") as f:
#         #     f.write(await file.read())
#         file_paths.append((file_path, file_name, candidate_id))
    
#     print("file path:", file_path)
#     print("file.filename:",file_name)


#     # Process resumes in parallel
#     results = []
#     user_email = request.session.get("user", "unknown")

#     with ThreadPoolExecutor(max_workers=4) as executor:
#         future_to_file = {
#             executor.submit(
#                 process_single_resume,
#                 file_path,
#                 file_name,
#                 job_description,
#                 MANDATORY_KEYWORDS,
#                 jd_id,
#                 user_email,
#             ): filename
#             for file_path, filename, _ in file_paths
#         }
#         for future in as_completed(future_to_file):
#             filename = future_to_file[future]
#             try:
#                 result = future.result()
#                 if result:
#                     results.append(result)
#             except Exception as exc:
#                 print(f"❌ Resume {filename} generated an exception: {exc}")

#     results = sorted(results, key=lambda x: x.get("final_score", 0), reverse=True)

#     # Cleanup
#     for file_path, _, _ in file_paths:
#         try:
#             os.remove(file_path)
#         except Exception as e:
#             print(f"⚠️ Failed to delete {file_path}: {e}")

#     total_time = time.time() - start_time
#     print(f"⚡ Processed {len(file_paths)} resumes in {total_time:.2f} seconds")

#     return templates.TemplateResponse(
#         "results.html",
#         {"request": request, "results": results, "es_resumes": resumes_from_es}
#     )







#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#BELOW WORKING DONT DELETE






import shutil
# @app.post("/process", response_class=HTMLResponse)
# async def process_resumes(
#     request: Request,
#     job_description: str = Form(...),
#     mandatory_keywords: str = Form(""),
# ):
#     start_time = time.time()

#     parsed_skills = extract_skills_from_jd(job_description)
#     print("parsed_skills:", parsed_skills)

#     # Collect keywords
#     mandatory_keywords_list = [kw.strip() for kw in mandatory_keywords.split(",") if kw.strip()]
#     MANDATORY_KEYWORDS = []
#     for key in [
#         "compulsory_programming_languages",
#         "must_cloud_services",
#         "must_databases",
#         "must_devops",
#         "must_big_data",
#         "must_framework",
#     ]:
#         if parsed_skills.get(key):
#             MANDATORY_KEYWORDS.extend(parsed_skills[key].split(","))
#     MANDATORY_KEYWORDS = [kw.strip() for kw in MANDATORY_KEYWORDS if kw.strip()]
#     MANDATORY_KEYWORDS = list(set(MANDATORY_KEYWORDS + mandatory_keywords_list))

#     # Save JD
#     try:
#         jd_id = save_job_description(job_description, created_by=request.session.get("user", "unknown"))
#     except Exception as e:
#         print(f"❌ Failed to save job description: {e}")
#         jd_id = None

#     # ✅ Call Elasticsearch
#     try:
#         es_handler = ElasticSearchHandler(
#             hostname="localhost",
#             port=9200,
#             scheme="http",
#             index_name="parsed_resumes_textfield"
#         )
#         resumes_from_es = es_handler.search_resumes(
#             min_exp_range=parsed_skills.get("min_exp_range", 0) or 0,
#             max_exp_range=parsed_skills.get("max_exp_range", 50) or 50,
#             compulsory_programming_languages=parsed_skills.get("compulsory_programming_languages", ""),
#             programming_languages_compulsory=parsed_skills.get("programming_languages_compulsory", False),
#             programming_languages=parsed_skills.get("programming_languages", ""),
#             cloud_services_compulsory=parsed_skills.get("cloud_services_compulsory", False),
#             must_cloud_services=parsed_skills.get("must_cloud_services", ""),
#             cloud_services=parsed_skills.get("cloud_services", ""),
#             databases_compulsory=parsed_skills.get("databases_compulsory", False),
#             must_databases=parsed_skills.get("must_databases", ""),
#             databases=parsed_skills.get("databases", ""),
#             devops_compulsory=parsed_skills.get("devops_compulsory", False),
#             must_devops=parsed_skills.get("must_devops", ""),
#             devops=parsed_skills.get("devops", ""),
#             big_data_compulsory=parsed_skills.get("big_data_compulsory", False),
#             must_big_data=parsed_skills.get("must_big_data", ""),
#             big_data=parsed_skills.get("big_data", ""),
#             framework_compulsory=parsed_skills.get("frameworks_compulsory", False),
#             must_framework=parsed_skills.get("must_framework", ""),
#             framework=parsed_skills.get("frameworks", "")
#         )
#         print("📂 Resumes from ES:", resumes_from_es)
#         resume_path = [details['resume_path'] for details in resumes_from_es.values()]

        
        
#         print("---------------------------------------------------")
#         print("resume path :", resume_path)
#         print("---------------------------------")
#     except Exception as e:
#         print(f"❌ Error fetching resumes from ES: {e}")
#         resumes_from_es = []
#         resume_path = []
    

#     # 📂 Permanent storage for resumes
#     permanent_dir = os.path.join(os.getcwd(), "uploaded_resumes")
#     os.makedirs(permanent_dir, exist_ok=True)

#     # Temp save uploaded resumes
#     temp_dir = tempfile.mkdtemp(prefix="resumes_")
#     file_paths = []

#     print("+++++++++++++++++++++++++++++++++++++++")
#     for file in resume_path or []:
#         print("file", file)
#         candidate_id = get_next_candidate_id()
#         file_path = file
#         file_name = os.path.basename(file_path)

#         # Copy resume into permanent folder
#         try:
#             permanent_path = os.path.join(permanent_dir, file_name)
#             shutil.copy(file_path, permanent_path)
#             print(f"✅ Stored {file_name} in uploaded_resumes/")
#         except Exception as e:
#             print(f"⚠️ Failed to copy {file_name} to uploaded_resumes: {e}")

#         file_paths.append((file_path, file_name, candidate_id))

#     print("file path:", file_path)
#     print("file.filename:", file_name)

#     # Process resumes in parallel
#     results = []
#     user_email = request.session.get("user", "unknown")

#     with ThreadPoolExecutor(max_workers=4) as executor:
#         future_to_file = {
#             executor.submit(
#                 process_single_resume,
#                 file_path,
#                 file_name,
#                 job_description,
#                 MANDATORY_KEYWORDS,
#                 jd_id,
#                 user_email,
#             ): filename
#             for file_path, filename, _ in file_paths
#         }
#         for future in as_completed(future_to_file):
#             filename = future_to_file[future]
#             try:
#                 result = future.result()
#                 if result:
#                     results.append(result)
#             except Exception as exc:
#                 print(f"❌ Resume {filename} generated an exception: {exc}")

#     results = sorted(results, key=lambda x: x.get("final_score", 0), reverse=True)

#     total_time = time.time() - start_time
#     print(f"⚡ Processed {len(file_paths)} resumes in {total_time:.2f} seconds")

#     return templates.TemplateResponse(
#         "results.html",
#         {"request": request, "results": results, "es_resumes": resumes_from_es}
#     )












#ABOVE WORKING DONT DELETE
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++








# @app.post("/process", response_class=HTMLResponse)
# async def process_resumes(
#     request: Request,
#     job_description: str = Form(...),
#     mandatory_keywords: str = Form(""),
# ):
#     start_time = time.time()

#     parsed_skills = extract_skills_from_jd(job_description)
#     print("parsed_skills:", parsed_skills)

#     # Collect keywords
#     mandatory_keywords_list = [kw.strip() for kw in mandatory_keywords.split(",") if kw.strip()]
#     MANDATORY_KEYWORDS = []
#     for key in [
#         "compulsory_programming_languages",
#         "must_cloud_services",
#         "must_databases",
#         "must_devops",
#         "must_big_data",
#         "must_framework",
#     ]:
#         if parsed_skills.get(key):
#             MANDATORY_KEYWORDS.extend(parsed_skills[key].split(","))
#     MANDATORY_KEYWORDS = [kw.strip() for kw in MANDATORY_KEYWORDS if kw.strip()]
#     MANDATORY_KEYWORDS = list(set(MANDATORY_KEYWORDS + mandatory_keywords_list))

#     # Save JD
#     try:
#         jd_id = save_job_description(job_description, created_by=request.session.get("user", "unknown"))
#     except Exception as e:
#         print(f"❌ Failed to save job description: {e}")
#         jd_id = None

#     # ✅ Call Elasticsearch
#     try:
#         es_handler = ElasticSearchHandler(
#             hostname="localhost",
#             port=9200,
#             scheme="http",
#             index_name="parsed_resumes_textfield"
#         )
#         resumes_from_es = es_handler.search_resumes(
#             min_exp_range=parsed_skills.get("min_exp_range", 0) or 0,
#             max_exp_range=parsed_skills.get("max_exp_range", 50) or 50,
#             compulsory_programming_languages=parsed_skills.get("compulsory_programming_languages", ""),
#             programming_languages_compulsory=parsed_skills.get("programming_languages_compulsory", False),
#             programming_languages=parsed_skills.get("programming_languages", ""),
#             cloud_services_compulsory=parsed_skills.get("cloud_services_compulsory", False),
#             must_cloud_services=parsed_skills.get("must_cloud_services", ""),
#             cloud_services=parsed_skills.get("cloud_services", ""),
#             databases_compulsory=parsed_skills.get("databases_compulsory", False),
#             must_databases=parsed_skills.get("must_databases", ""),
#             databases=parsed_skills.get("databases", ""),
#             devops_compulsory=parsed_skills.get("devops_compulsory", False),
#             must_devops=parsed_skills.get("must_devops", ""),
#             devops=parsed_skills.get("devops", ""),
#             big_data_compulsory=parsed_skills.get("big_data_compulsory", False),
#             must_big_data=parsed_skills.get("must_big_data", ""),
#             big_data=parsed_skills.get("big_data", ""),
#             framework_compulsory=parsed_skills.get("frameworks_compulsory", False),
#             must_framework=parsed_skills.get("must_framework", ""),
#             framework=parsed_skills.get("frameworks", "")
#         )
#         print("📂 Resumes from ES:", resumes_from_es)
#         resume_path = [details['resume_path'] for details in resumes_from_es.values()]

#         print("---------------------------------------------------")
#         print("Resume path :", resume_path)
#         print("---------------------------------")
#     except Exception as e:
#         print(f"❌ Error fetching resumes from ES: {e}")
#         resumes_from_es = {}
#         resume_path = []

#     # 📂 Permanent storage for resumes
#     permanent_dir = os.path.join(os.getcwd(), "uploaded_resumes")
#     os.makedirs(permanent_dir, exist_ok=True)

#     temp_dir = tempfile.mkdtemp(prefix="resumes_")
#     file_paths = []

#     print("+++++++++++++++++++++++++++++++++++++++")
#     for file in resume_path or []:
#         print("file", file)
#         candidate_id = get_next_candidate_id()
#         file_path = file
#         file_name = os.path.basename(file_path)

#         try:
#             permanent_path = os.path.join(permanent_dir, file_name)
#             shutil.copy(file_path, permanent_path)
#             print(f"✅ Stored {file_name} in uploaded_resumes/")
#         except Exception as e:
#             print(f"⚠️ Failed to copy {file_name} to uploaded_resumes: {e}")

#         file_paths.append((file_path, file_name, candidate_id))

#     if not file_paths:
#         print("⚠️ No resumes found to process.")
#         return templates.TemplateResponse(
#             "results.html",
#             {
#                 "request": request,
#                 "results": [],
#                 "es_resumes": resumes_from_es,
#                 "message": f"No resumes found for job description ID: {jd_id if jd_id else 'N/A'}"
#             }
#         )

#     # Process resumes in parallel
#     results = []
#     user_email = request.session.get("user", "unknown")

#     with ThreadPoolExecutor(max_workers=4) as executor:
#         future_to_file = {
#             executor.submit(
#                 process_single_resume,
#                 file_path,
#                 file_name,
#                 job_description,
#                 MANDATORY_KEYWORDS,
#                 jd_id,
#                 user_email,
#             ): filename
#             for file_path, filename, _ in file_paths
#         }
#         for future in as_completed(future_to_file):
#             filename = future_to_file[future]
#             try:
#                 result = future.result()
#                 if result:
#                     results.append(result)
#             except Exception as exc:
#                 print(f"❌ Resume {filename} generated an exception: {exc}")

#     results = sorted(results, key=lambda x: x.get("final_score", 0), reverse=True)

#     total_time = time.time() - start_time
#     print(f"⚡ Processed {len(file_paths)} resumes in {total_time:.2f} seconds")

#     return templates.TemplateResponse(
#         "results.html",
#         {"request": request, "results": results, "es_resumes": resumes_from_es}
#     )


























@app.post("/process", response_class=HTMLResponse)
async def process_resumes(
    request: Request,
    job_description: str = Form(...),
    mandatory_keywords: str = Form(""),
):
    start_time = time.time()

    parsed_skills = extract_skills_from_jd(job_description)
    print("parsed_skills:", parsed_skills)

    # Collect keywords
    mandatory_keywords_list = [kw.strip() for kw in mandatory_keywords.split(",") if kw.strip()]
    MANDATORY_KEYWORDS = []
    for key in [
        "compulsory_programming_languages",
        "must_cloud_services",
        "must_databases",
        "must_devops",
        "must_big_data",
        "must_framework",
    ]:
        if parsed_skills.get(key):
            MANDATORY_KEYWORDS.extend(parsed_skills[key].split(","))
    MANDATORY_KEYWORDS = [kw.strip() for kw in MANDATORY_KEYWORDS if kw.strip()]
    MANDATORY_KEYWORDS = list(set(MANDATORY_KEYWORDS + mandatory_keywords_list))

    # Save JD
    try:
        jd_id = save_job_description(job_description, created_by=request.session.get("user", "unknown"))
    except Exception as e:
        print(f"❌ Failed to save job description: {e}")
        jd_id = None

    # ✅ Call Elasticsearch
    try:
        es_handler = ElasticSearchHandler(
            hostname="localhost",
            port=9200,
            scheme="http",
            index_name="parsed_resumes_textfield"
        )
        resumes_from_es = es_handler.search_resumes(
            min_exp_range=parsed_skills.get("min_exp_range", 0) or 0,
            max_exp_range=parsed_skills.get("max_exp_range", 50) or 50,
            compulsory_programming_languages=parsed_skills.get("compulsory_programming_languages", ""),
            programming_languages_compulsory=parsed_skills.get("programming_languages_compulsory", False),
            programming_languages=parsed_skills.get("programming_languages", ""),
            cloud_services_compulsory=parsed_skills.get("cloud_services_compulsory", False),
            must_cloud_services=parsed_skills.get("must_cloud_services", ""),
            cloud_services=parsed_skills.get("cloud_services", ""),
            databases_compulsory=parsed_skills.get("databases_compulsory", False),
            must_databases=parsed_skills.get("must_databases", ""),
            databases=parsed_skills.get("databases", ""),
            devops_compulsory=parsed_skills.get("devops_compulsory", False),
            must_devops=parsed_skills.get("must_devops", ""),
            devops=parsed_skills.get("devops", ""),
            big_data_compulsory=parsed_skills.get("big_data_compulsory", False),
            must_big_data=parsed_skills.get("must_big_data", ""),
            big_data=parsed_skills.get("big_data", ""),
            framework_compulsory=parsed_skills.get("frameworks_compulsory", False),
            must_framework=parsed_skills.get("must_framework", ""),
            framework=parsed_skills.get("frameworks", "")
        )
        print("📂 Resumes from ES:", resumes_from_es)
        resume_path = [details['resume_path'] for details in resumes_from_es.values()]

        print("---------------------------------------------------")
        print("Resume path :", resume_path)
        print("---------------------------------")
    except Exception as e:
        print(f"❌ Error fetching resumes from ES: {e}")
        resumes_from_es = {}
        resume_path = []

    # 📂 Permanent storage for resumes
    permanent_dir = os.path.join(os.getcwd(), "uploaded_resumes")
    os.makedirs(permanent_dir, exist_ok=True)

    temp_dir = tempfile.mkdtemp(prefix="resumes_")
    file_paths = []

    print("+++++++++++++++++++++++++++++++++++++++")
    for file in resume_path or []:
        print("file", file)
        candidate_id = get_next_candidate_id()
        file_path = file
        file_name = os.path.basename(file_path)

        try:
            permanent_path = os.path.join(permanent_dir, file_name)
            if os.path.abspath(file_path) != os.path.abspath(permanent_path):
                shutil.copy(file_path, permanent_path)
                print(f"✅ Stored {file_name} in uploaded_resumes/")
            else:
                print(f"ℹ️ Skipped copying {file_name}, already in uploaded_resumes.")
        except Exception as e:
            print(f"⚠️ Failed to copy {file_name} to uploaded_resumes: {e}")

        file_paths.append((file_path, file_name, candidate_id))

    # ⚠️ Handle no resumes found
    if not file_paths:
        print("⚠️ No resumes found to process.")
        return templates.TemplateResponse(
            "results.html",
            {
                "request": request,
                "results": [],
                "es_resumes": resumes_from_es,
                "message": f"No resumes found for job description ID: {jd_id if jd_id else 'N/A'}"
            }
        )

    # Process resumes in parallel
    results = []
    user_email = request.session.get("user", "unknown")

    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_file = {
            executor.submit(
                process_single_resume,
                file_path,
                file_name,
                job_description,
                MANDATORY_KEYWORDS,
                jd_id,
                user_email,
            ): filename
            for file_path, filename, _ in file_paths
        }
        for future in as_completed(future_to_file):
            filename = future_to_file[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as exc:
                print(f"❌ Resume {filename} generated an exception: {exc}")

    results = sorted(results, key=lambda x: x.get("final_score", 0), reverse=True)

    total_time = time.time() - start_time
    print(f"⚡ Processed {len(file_paths)} resumes in {total_time:.2f} seconds")

    return templates.TemplateResponse(
        "results.html",
        {"request": request, "results": results, "es_resumes": resumes_from_es}
    )




























# ABOVE WORKING



















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
import time

# @app.post("/process", response_class=HTMLResponse)
# async def process_resumes(
#     request: Request,
#     job_description: str = Form(...),
#     mandatory_keywords: str = Form(...),
#     uploaded_resumes: List[UploadFile] = None
# ):
#     start_time = time.time()  # ⏱️ Start timer

#     MANDATORY_KEYWORDS = [kw.strip() for kw in mandatory_keywords.split(",") if kw.strip()]
#     results = []

#     jd_id = save_job_description(job_description, created_by=request.session.get("user", "unknown"))

#     UPLOAD_FOLDER = "uploads"
#     os.makedirs(UPLOAD_FOLDER, exist_ok=True)

#     for file in uploaded_resumes or []:
#         candidate_id = get_next_candidate_id()

#         file_path = os.path.join(UPLOAD_FOLDER, file.filename)
#         with open(file_path, "wb") as f:
#             f.write(await file.read())

#         resume_text = get_resume_text(file_path)
#         if not resume_text:
#             continue

#         resume_text = resume_text.replace("\n", " ").strip()
#         candidate_name, projects = extract_candidate_info_with_projects(resume_text)

#         keyword_check_result = check_mandatory_keywords_in_projects(projects, MANDATORY_KEYWORDS)
#         missing_skills = extract_missing_skills(keyword_check_result)
#         missing_keywords_note = ', '.join(missing_skills) if missing_skills else ""

#         project_levels = classify_project_levels(projects)
#         project_skill_df = build_project_skill_matrix(projects, MANDATORY_KEYWORDS)
#         skill_scores, total_score = calculate_project_skill_score(project_skill_df)

#         missing_keywords = [kw for kw in MANDATORY_KEYWORDS if kw.lower() not in resume_text.lower()]
#         missing_note = f"Missing Keywords: {', '.join(missing_keywords)}" if missing_keywords else ""

#         jd_exp = extract_required_experience(job_description)
#         resume_exp_text = extract_experience(resume_text)
#         resume_exp = parse_resume_experience(resume_exp_text)
#         if jd_exp > 0:
#             exp_warning = (
#                 f" Candidate has {resume_exp} years, JD requires {jd_exp} years"
#                 if resume_exp < jd_exp
#                 else f"✅ Candidate meets requirement ({resume_exp} vs {jd_exp} years)"
#             )
#         else:
#             exp_warning = "ℹ️ JD has no explicit experience requirement"

#         cos_score = cosine_similarity_score(job_description, resume_text) * 100
#         genai_score, genai_reason = llm_score_with_reason(job_description, resume_text)
#         authenticity_report = llm_experience_verification(resume_text)

#         final_score = round((cos_score * 0.3) + (genai_score * 0.7), 2)

#         save_candidate_report(
#             candidate_id=candidate_id,
#             candidate_name=candidate_name,
#             resume_filename=file.filename,
#             experience=resume_exp,
#             experience_details=resume_exp_text,
#             cosine_score=round(cos_score, 2),
#             genai_score=round(genai_score, 2),
#             final_score=final_score,
#             missing_keywords_note=", ".join(missing_skills) if missing_skills else "",
#             exp_warning=exp_warning,
#             projects_list=projects,
#             project_skill_df=project_skill_df.to_dict(orient="records"),
#             mandatory_skill_check_df=keyword_check_result,
#             project_level_df=project_levels,
#             individual_skill_scores=skill_scores,
#             genai_reason=genai_reason,
#             authenticity_report=authenticity_report,
#             jd_id=jd_id,
#             created_by=request.session.get("user", "unknown"),
#             updated_by=request.session.get("user", "unknown")
#         )

#         results.append({
#             "filename": file.filename,
#             "candidate_name": candidate_name,
#             "experience": resume_exp,
#             "experience_details": resume_exp_text,
#             "cosine_similarity": round(cos_score, 2),
#             "genai_score": round(genai_score, 2),
#             "final_score": final_score,
#             "missing_keywords_note": missing_note,
#             "projects": projects,
#             "exp_warning": exp_warning,
#             "mandatory_keywords_html": keyword_check_result,
#             "project_skill_html": project_skill_df.to_html(classes="table table-striped", index=True),
#             "project_levels_html": pd.DataFrame(project_levels).to_html(classes="table table-striped", index=False),
#             "skill_scores": skill_scores,
#             "total_skill_score": total_score,
#             "genai_reason": genai_reason,
#             "authenticity_report": authenticity_report
#         })

#     results = sorted(results, key=lambda x: x["final_score"], reverse=True)

#     total_time = time.time() - start_time  # ⏱️ End timer
#     print(f"⚡ Total processing time: {total_time:.2f} seconds for {len(uploaded_resumes or [])} resumes")  

#     return templates.TemplateResponse("results.html", {"request": request, "results": results})




# @app.post("/process", response_class=HTMLResponse)
# async def process_resumes(
#     request: Request,
#     job_description: str = Form(...),
#     uploaded_resumes: List[UploadFile] = None
# ):
#     start_time = time.time()  #  Start timer

#     # ✅ Extract skills from JD
#     parsed_skills = extract_skills_from_jd(job_description)

#     # Collect mandatory keywords (you can tweak logic here)
#     MANDATORY_KEYWORDS = []
#     if parsed_skills.get("compulsory_programming_languages"):
#         MANDATORY_KEYWORDS.extend(parsed_skills["compulsory_programming_languages"].split(","))
#     if parsed_skills.get("must_cloud_services"):
#         MANDATORY_KEYWORDS.extend(parsed_skills["must_cloud_services"].split(","))
#     if parsed_skills.get("must_databases"):
#         MANDATORY_KEYWORDS.extend(parsed_skills["must_databases"].split(","))
#     if parsed_skills.get("must_devops"):
#         MANDATORY_KEYWORDS.extend(parsed_skills["must_devops"].split(","))
#     if parsed_skills.get("must_big_data"):
#         MANDATORY_KEYWORDS.extend(parsed_skills["must_big_data"].split(","))
#     if parsed_skills.get("must_framework"):
#         MANDATORY_KEYWORDS.extend(parsed_skills["must_framework"].split(","))

#     # Strip and clean keywords
#     MANDATORY_KEYWORDS = [kw.strip() for kw in MANDATORY_KEYWORDS if kw.strip()]

#     # Save job description
#     jd_id = save_job_description(job_description, created_by=request.session.get("user", "unknown"))
    
#     UPLOAD_FOLDER = "uploads"
#     os.makedirs(UPLOAD_FOLDER, exist_ok=True)

#     # Save all files first
#     file_paths = []
#     for file in uploaded_resumes or []:
#         candidate_id = get_next_candidate_id()
#         file_path = os.path.join(UPLOAD_FOLDER, file.filename)
#         with open(file_path, "wb") as f:
#             f.write(await file.read())
#         file_paths.append((file_path, file.filename))

#     # Process resumes in parallel
#     results = []
#     user_email = request.session.get("user", "unknown")
    
#     with ThreadPoolExecutor(max_workers=4) as executor:  
#         future_to_file = {
#             executor.submit(
#                 process_single_resume, 
#                 file_path, 
#                 filename, 
#                 job_description, 
#                 MANDATORY_KEYWORDS, 
#                 jd_id, 
#                 user_email
#             ): filename 
#             for file_path, filename in file_paths
#         }
        
#         for future in as_completed(future_to_file):
#             filename = future_to_file[future]
#             try:
#                 result = future.result()
#                 if result:
#                     results.append(result)
#             except Exception as exc:
#                 print(f'❌ Resume {filename} generated an exception: {exc}')

#     # Sort by final_score
#     results = sorted(results, key=lambda x: x["final_score"], reverse=True)
    
#     # Clean up uploaded files
#     for file_path, _ in file_paths:
#         try:
#             os.remove(file_path)
#         except:
#             pass

#     total_time = time.time() - start_time  # ⏱️ End timer
#     print(f"⚡ Processed {len(file_paths)} resumes in {total_time:.2f} seconds using ThreadPoolExecutor")

#     return templates.TemplateResponse("results.html", {"request": request, "results": results})











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
















# @app.post("/process_resume", response_class=HTMLResponse)
# async def process_resumes(
#     request: Request,
#     job_description: str = Form(...),
#     mandatory_keywords: str = Form(...),
#     uploaded_resumes: List[UploadFile] = None
# ):
#     MANDATORY_KEYWORDS = [kw.strip() for kw in mandatory_keywords.split(",") if kw.strip()]
    
#     # Save job description
#     jd_id = save_job_description(job_description, created_by=request.session.get("user", "unknown"))
    
#     UPLOAD_FOLDER = "uploads"
#     os.makedirs(UPLOAD_FOLDER, exist_ok=True)
 
#     # Save all files first
#     file_paths = []
#     for file in uploaded_resumes or []:
#         file_path = os.path.join(UPLOAD_FOLDER, file.filename)
#         with open(file_path, "wb") as f:
#             f.write(await file.read())
#         file_paths.append((file_path, file.filename))
 
#     # Process resumes in parallel
#     results = []
#     user_email = request.session.get("user", "unknown")
    
#     # Use ThreadPoolExecutor for parallel processing
#     with ThreadPoolExecutor(max_workers=4) as executor:  # Adjust max_workers based on your system
#         # Submit all tasks
#         future_to_file = {
#             executor.submit(
#                 process_single_resume,
#                 file_path,
#                 filename,
#                 job_description,
#                 MANDATORY_KEYWORDS,
#                 jd_id,
#                 user_email
#             ): filename
#             for file_path, filename in file_paths
#         }
        
#         # Collect results as they complete
#         for future in as_completed(future_to_file):
#             filename = future_to_file[future]
#             try:
#                 result = future.result()
#                 if result:  # Only add successful results
#                     results.append(result)
#             except Exception as exc:
#                 print(f'Resume {filename} generated an exception: {exc}')
 
#     # Sort by final_score
#     results = sorted(results, key=lambda x: x["final_score"], reverse=True)
    
#     # Clean up uploaded files (optional)
#     for file_path, _ in file_paths:
#         try:
#             os.remove(file_path)
#         except:
#             pass  # Ignore cleanup errors
    
#     return templates.TemplateResponse("results.html", {"request": request, "results": results})
 







 





















from fastapi import Request, Form, UploadFile
from fastapi.responses import HTMLResponse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from typing import List

@app.post("/process_resume", response_class=HTMLResponse)
async def process_resumes(
    request: Request,
    job_description: str = Form(...),
    mandatory_keywords: str = Form(...),
    uploaded_resumes: List[UploadFile] = None
):
    MANDATORY_KEYWORDS = [kw.strip() for kw in mandatory_keywords.split(",") if kw.strip()]

    # Save job description
    jd_id = save_job_description(job_description, created_by=request.session.get("user", "unknown"))

    # Permanent storage folder for resumes
    PERMANENT_UPLOAD_FOLDER = "uploaded_resumes"
    os.makedirs(PERMANENT_UPLOAD_FOLDER, exist_ok=True)

    # Temporary folder for processing
    TEMP_UPLOAD_FOLDER = "uploads"
    os.makedirs(TEMP_UPLOAD_FOLDER, exist_ok=True)

    # Save all files
    file_paths = []
    for file in uploaded_resumes or []:
        temp_path = os.path.join(TEMP_UPLOAD_FOLDER, file.filename)
        permanent_path = os.path.join(PERMANENT_UPLOAD_FOLDER, file.filename)

        # Save to temp folder (for processing)
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        # Also store a permanent copy
        with open(permanent_path, "wb") as f:
            f.write(await file.read())

        file_paths.append((temp_path, file.filename))

    # Process resumes in parallel
    results = []
    user_email = request.session.get("user", "unknown")

    with ThreadPoolExecutor(max_workers=4) as executor:
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

        for future in as_completed(future_to_file):
            filename = future_to_file[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as exc:
                print(f'Resume {filename} generated an exception: {exc}')

    results = sorted(results, key=lambda x: x["final_score"], reverse=True)

    # Clean up only temporary files
    for file_path, _ in file_paths:
        try:
            os.remove(file_path)
        except:
            pass

    return templates.TemplateResponse("results.html", {"request": request, "results": results})