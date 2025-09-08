from pymongo import MongoClient
from datetime import datetime
from werkzeug.security import generate_password_hash

from pymongo.errors import DuplicateKeyError


# MongoDB connection
client = MongoClient("mongodb://localhost:27017/")
db = client["resume_screening"]

candidates_collection = db["candidates"]
jd_collection = db["job_descriptions"]
users_collection = db["users"]

status_collection = db["candidate_status"]


# ------------------------------
# Users Collection
# ------------------------------

# Generate unique user IDs
def get_next_user_id():
    counter = db.counters.find_one_and_update(
        {"_id": "user_id"},
        {"$inc": {"seq": 1}},
        upsert=True,
        return_document=True
    )
    return f"USER{counter['seq']:04d}"   # e.g., USER0001

# Save a new user
def save_user(user_id, name, email, password, role="recruiter", created_by="system"):
    document = {
        "user_id": user_id,
        "name": name,
        "email": email.lower(),
        "password": generate_password_hash(password, method="pbkdf2:sha256"),  # hashes here!
        "role": role,
        "created_by": created_by,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    users_collection.insert_one(document)


# Fetch a user by email
def get_user_by_email(email):
    """
    Retrieve user by email (used for login).
    """
    return users_collection.find_one({"email": email.lower()})


# ------------------------------
# candidate Collection
# ------------------------------

def get_next_candidate_id():
    counter = db.counters.find_one_and_update(
        {"_id": "candidate_id"},
        {"$inc": {"seq": 1}},
        upsert=True,
        return_document=True
    )
    return f"CAND{counter['seq']:04d}"


# ------------------------------
# JD Collection
# ------------------------------

def get_next_jd_id():
    counter = db.counters.find_one_and_update(
        {"_id": "jd_id"},
        {"$inc": {"seq": 1}},
        upsert=True,
        return_document=True
    )
    return f"JD{counter['seq']:03d}"



def save_candidate_report(resume_filename,candidate_id, candidate_name, experience, experience_details,
                          cosine_score, genai_score, final_score,
                          missing_keywords_note, exp_warning,
                          projects_list, project_skill_df,
                          mandatory_skill_check_df, project_level_df,
                          individual_skill_scores, genai_reason,jd_id, created_by, updated_by,
                          authenticity_report):
    
    document = {
        "candidate_id": candidate_id,
        "candidate_name": candidate_name,
        "filename": resume_filename,
        "experience": experience,
        "experience_details": experience_details,
        "cosine_similarity": cosine_score,
        "genai_score": genai_score,
        "final_score": final_score,
        "missing_keywords_note": missing_keywords_note,
        "exp_warning": exp_warning,
        "projects": projects_list,
        "project_skill_match": project_skill_df,
        "mandatory_skill_check": mandatory_skill_check_df,
        "project_complexity_levels": project_level_df,
        "individual_skill_scores": individual_skill_scores,
        "genai_reason": genai_reason,
        "authenticity_report": authenticity_report,
        "jd_id": jd_id,                  # Link to JD
        "created_by": created_by,        # User/system who created entry
        "updated_by": updated_by,        # Last user/system updated
        "created_at": datetime.utcnow()
        # "updated_at": datetime.utcnow()
    }
    
    candidates_collection.insert_one(document)


# ------------------------------
# Save Job Description
# ------------------------------


def save_job_description(jd_description, created_by="admin_user"):
    # First check if JD already exists
    existing_jd = jd_collection.find_one({"jd_description": jd_description})
    
    if existing_jd:
        print(f"JD already exists with ID: {existing_jd['jd_id']}")
        return existing_jd['jd_id']
    
    # If not exists, generate a new jd_id
    jd_id = get_next_jd_id()
    document = {
        "jd_id": jd_id,
        "jd_description": jd_description,
        "created_by": created_by,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    
    try:
        jd_collection.insert_one(document)
        print(f"New JD inserted with ID: {jd_id}")
        return jd_id
    except DuplicateKeyError:
        # In case of race condition (two inserts at same time)
        existing_jd = jd_collection.find_one({"jd_description": jd_description})
        print(f"JD already exists with ID: {existing_jd['jd_id']}")
        return existing_jd['jd_id']
    
        

def save_candidate_status(jd_id, candidate_id, status, updated_by="system"):
    """
    Save or update candidate status for a specific JD.
    status: "accept", "hold", "reject"
    """
    # Use update_one with upsert=True so it inserts if no document exists
    status_collection.update_one(
        {"jd_id": jd_id, "candidate_id": candidate_id},  # filter
        {"$set": {
            "status": status,
            "updated_by": updated_by,
            "updated_at": datetime.utcnow()
        }},
        upsert=True
    )