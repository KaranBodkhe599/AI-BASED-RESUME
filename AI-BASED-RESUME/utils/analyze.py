# utils/analyze.py
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Example job description (replace with your real job requirements)
JOB_DESCRIPTION = """
Python, SQL, Machine Learning, Data Analysis, AWS, Docker, React, NLP, Deep Learning
"""

JOB_SKILLS = ["Python", "SQL", "Machine Learning", "Data Analysis", "AWS", "Docker", "React", "NLP", "Deep Learning"]

EDUCATION_LEVELS = {
    "phd": 5,
    "doctorate": 5,
    "master": 4,
    "bachelor": 3,
    "diploma": 2,
    "high school": 1,
    "secondary": 1
}

def analyze_resume(resume_text):
    """
    Analyze resume text against JOB_DESCRIPTION and JOB_SKILLS.
    Returns a dictionary with all dynamic fields.
    """

    text = resume_text.lower()

    # -------------------------
    # 1️⃣ Similarity Score
    # -------------------------
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([JOB_DESCRIPTION.lower(), text])
    similarity_score = round(cosine_similarity(vectors[0:1], vectors[1:2])[0][0] * 100, 2)

    # -------------------------
    # 2️⃣ Skills Match
    # -------------------------
    matched_skills = [skill for skill in JOB_SKILLS if skill.lower() in text]
    missing_skills = [skill for skill in JOB_SKILLS if skill.lower() not in text]
    matched_count = len(matched_skills)
    missing_count = len(missing_skills)

    # -------------------------
    # 3️⃣ Years of Experience
    # -------------------------
    exp_matches = re.findall(r'(\d+)\s+years?', text)
    years_experience = max([int(x) for x in exp_matches], default=0)

    # -------------------------
    # 4️⃣ Education Level
    # -------------------------
    education_level = 0
    for keyword, level in EDUCATION_LEVELS.items():
        if keyword in text:
            education_level = max(education_level, level)

    # -------------------------
    # 5️⃣ Overall Score
    # -------------------------
    overall_score = round(
        (similarity_score + 
         (matched_count / len(JOB_SKILLS) * 100) +
         min(years_experience * 10, 100) + 
         education_level * 20) / 4
    )

    return {
        "overall_score": overall_score,
        "similarity_score": similarity_score,
        "matched_count": matched_count,
        "missing_count": missing_count,
        "years_experience": years_experience,
        "education_level": education_level,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills
    }
