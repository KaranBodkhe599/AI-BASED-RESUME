from flask import Flask, render_template, request, redirect, url_for, session, flash
import sqlite3
import os
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from utils.extract import extract_text
from utils.preprocess import clean_text
from sentence_transformers import SentenceTransformer, util

# ----------------- FLASK CONFIG -----------------
app = Flask(__name__)
app.secret_key = "super_secret_key"
UPLOAD_FOLDER = "resumes"
ALLOWED_EXTENSIONS = {'pdf', 'docx'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# ----------------- SENTENCE-BERT MODEL -----------------
print("Loading Sentence-BERT model...")
model = SentenceTransformer('all-mpnet-base-v2')
print("Model loaded successfully!")

# ----------------- DATABASE SETUP -----------------
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()

    # Create users table if not exists
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)

    # Create scans table if not exists
    c.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            filename TEXT,
            similarity_score REAL,
            matched_skills TEXT,
            missing_skills TEXT,
            experience REAL DEFAULT 0,
            education_level REAL DEFAULT 0,
            scan_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Ensure experience and education_level columns exist (in case of old tables)
    try:
        c.execute("ALTER TABLE scans ADD COLUMN experience REAL DEFAULT 0")
    except sqlite3.OperationalError:
        pass  # Column already exists

    try:
        c.execute("ALTER TABLE scans ADD COLUMN education_level REAL DEFAULT 0")
    except sqlite3.OperationalError:
        pass  # Column already exists

    conn.commit()
    conn.close()

init_db()

# ----------------- HELPER FUNCTIONS -----------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

SKILL_DEFINITIONS = {
    "python": {"variations": ["python"], "contexts": ["experience", "worked", "developed"], "exclusions": []},
    "java": {"variations": ["java"], "contexts": ["experience", "worked", "developed"], "exclusions": ["javascript"]},
    "javascript": {"variations": ["javascript", "js"], "contexts": ["experience", "worked", "developed"], "exclusions": []},
    "html": {"variations": ["html"], "contexts": ["experience", "worked", "developed"], "exclusions": []},
    "css": {"variations": ["css"], "contexts": ["experience", "worked", "developed"], "exclusions": []},
}

def extract_skills_from_text(text, skill_definitions):
    text_lower = text.lower()
    matched_skills = []
    missing_skills = []

    for skill_name, skill_info in skill_definitions.items():
        found = any(var in text_lower for var in skill_info["variations"])
        if found:
            matched_skills.append(skill_name)
        else:
            missing_skills.append(skill_name)

    return matched_skills, missing_skills

def calculate_semantic_similarity(job_desc, resume_text):
    job_embedding = model.encode(job_desc, convert_to_tensor=True)
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    cosine_score = util.pytorch_cos_sim(job_embedding, resume_embedding)
    return cosine_score.item() * 100  # Return as percentage

# ----------------- ROUTES -----------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if not username or not password:
            flash("Username and password required!", "error")
            return render_template("register.html")
        password_hash = generate_password_hash(password)
        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password_hash))
            conn.commit()
            flash("Registration successful! Please login.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Username already exists!", "error")
        finally:
            conn.close()
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("SELECT id, password FROM users WHERE username=?", (username,))
        user = c.fetchone()
        conn.close()
        if user and check_password_hash(user[1], password):
            session["user_id"] = user[0]
            session["username"] = username
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid username or password!", "error")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

@app.route("/dashboard")
def dashboard():
    if "username" not in session:
        return redirect(url_for("login"))
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("""
        SELECT filename, similarity_score, scan_date, matched_skills, missing_skills, experience, education_level
        FROM scans 
        WHERE user_id = ? 
        ORDER BY scan_date DESC 
        LIMIT 10
    """, (session["user_id"],))
    scans = c.fetchall()
    conn.close()
    return render_template("dashboard.html", username=session["username"], scans=scans)

@app.route("/upload", methods=["POST"])
def upload_resume():
    if "username" not in session:
        return redirect(url_for("login"))

    job_desc = request.form.get("jobdesc", "").strip()
    resume = request.files.get("resume")
    
    if not job_desc or not resume or resume.filename == '':
        flash("Job description and resume required!", "error")
        return redirect(url_for("dashboard"))
    
    if not allowed_file(resume.filename):
        flash("Invalid file type! Only PDF/DOCX allowed.", "error")
        return redirect(url_for("dashboard"))

    filename = secure_filename(resume.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    resume.save(file_path)

    # Extract and preprocess text
    resume_text = extract_text(file_path)
    resume_clean = clean_text(resume_text)
    job_clean = clean_text(job_desc)

    # Semantic similarity
    similarity = calculate_semantic_similarity(job_clean, resume_clean)

    # Skill extraction
    matched_skills, missing_skills = extract_skills_from_text(resume_text, SKILL_DEFINITIONS)
    matched_count = len(matched_skills)
    missing_count = len(missing_skills)

    # Extract experience & education (simple placeholder extraction)
    years_experience = 0
    education_level = 0
    lines = resume_text.lower().split("\n")
    for line in lines:
        if "years" in line:
            try:
                years_experience = int(''.join(filter(str.isdigit, line)))
            except:
                pass
        if "bachelor" in line or "graduate" in line:
            education_level = 1
        elif "master" in line:
            education_level = 2
        elif "phd" in line:
            education_level = 3

    # Save scan to DB
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("""
        INSERT INTO scans (user_id, filename, similarity_score, matched_skills, missing_skills, experience, education_level)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        session["user_id"],
        filename,
        round(similarity, 2),
        ','.join(matched_skills),
        ','.join(missing_skills),
        years_experience,
        education_level
    ))
    conn.commit()
    conn.close()

    os.remove(file_path)

    # Render result template with proper variables
    return render_template("result.html",
                           filename=filename,
                           similarity_score=round(similarity, 2),
                           score=round(similarity, 2),
                           matched_skills=matched_skills,
                           missing_skills=missing_skills,
                           matched_count=matched_count,
                           missing_count=missing_count,
                           years_experience=years_experience,
                           education_level=education_level)

# ----------------- RUN APP -----------------
if __name__ == "__main__":
    app.run(debug=True)
