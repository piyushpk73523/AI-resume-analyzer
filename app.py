
from flask import Flask, render_template, request
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os, re

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text.lower()

def extract_skills(text):
    skills = [
        "python","java","javascript","react","node","express","mongodb",
        "sql","machine learning","deep learning","nlp","flask","django",
        "git","docker","aws","linux","rest api"
    ]
    found = set()
    for skill in skills:
        if re.search(r"\b" + re.escape(skill) + r"\b", text):
            found.add(skill)
    return found

def calculate_similarity(resume_text, jd_text):
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform([resume_text, jd_text])
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])
    return round(similarity[0][0] * 100, 2)

@app.route("/", methods=["GET","POST"])
def index():
    score = None
    missing_skills = []

    if request.method == "POST":
        resume = request.files["resume"]
        jd = request.form["jd"]

        if resume:
            path = os.path.join(app.config["UPLOAD_FOLDER"], resume.filename)
            resume.save(path)

            resume_text = extract_text_from_pdf(path)
            jd_text = jd.lower()

            score = calculate_similarity(resume_text, jd_text)

            resume_skills = extract_skills(resume_text)
            jd_skills = extract_skills(jd_text)

            missing_skills = sorted(list(jd_skills - resume_skills))

    return render_template("index.html", score=score, missing_skills=missing_skills)

if __name__ == "__main__":
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    app.run(debug=True)
