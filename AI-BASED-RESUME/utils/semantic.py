from sentence_transformers import SentenceTransformer, util
from utils.preprocess import clean_text
import re

# Skill keywords
SKILL_KEYWORDS = [
    "python", "flask", "django", "aws", "sql",
    "javascript", "html", "css"
]

# Load model once (fast + GPU optimized)
model = SentenceTransformer("all-mpnet-base-v2")


def find_missing_skills(text):
    """
    Identify missing skills using word-boundary matching.
    """
    text = text.lower()

    missing = []
    for skill in SKILL_KEYWORDS:
        pattern = rf"\b{re.escape(skill)}\b"
        if not re.search(pattern, text):
            missing.append(skill)

    return missing


def get_similarity(resume_text, job_desc):
    """
    Returns similarity score (0–1 range) and missing skills.
    """
    # Clean text
    resume_clean = clean_text(resume_text)
    job_clean = clean_text(job_desc)

    # Encode both texts
    embeddings = model.encode([resume_clean, job_clean], convert_to_tensor=True)

    # Cosine similarity
    score = float(util.cos_sim(embeddings[0], embeddings[1]))

    # Fix negative range: convert raw cosine (-1 to 1) → (0 to 1)
    normalized_score = (score + 1) / 2

    # Detect missing skills
    missing_skills = find_missing_skills(resume_clean)

    return normalized_score, missing_skills
