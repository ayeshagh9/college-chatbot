from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "college_data.csv")

# Load dataset safely
try:
    data = pd.read_csv(CSV_PATH, encoding="utf-8", error_bad_lines=False)
except Exception:
    data = pd.read_csv(CSV_PATH)

# Normalize column names
data.columns = [col.strip().lower() for col in data.columns]

# Auto-fix simple column issues
if "question" not in data.columns or "answer" not in data.columns:
    if len(data.columns) >= 3:
        data.columns = ["category", "question", "answer"]
    elif len(data.columns) == 2:
        data.columns = ["question", "answer"]
    else:
        raise ValueError("CSV format is incorrect. Expected at least question and answer columns.")

# Remove empty rows
data = data.dropna(subset=["question", "answer"])

questions = data["question"].astype(str).str.strip().tolist()
answers = data["answer"].astype(str).str.strip().tolist()

def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

cleaned_questions = [clean_text(q) for q in questions]

vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(cleaned_questions)

college_keywords = {
    "admission", "admissions", "apply", "course", "courses", "fee", "fees",
    "exam", "exams", "hallticket", "hall", "ticket", "attendance", "hostel",
    "library", "placement", "placements", "syllabus", "faculty", "department",
    "college", "university", "campus", "transport", "scholarship", "results",
    "revaluation", "certificate", "bonafide", "tc", "id", "principal", "office",
    "contact", "sports", "ncc", "nss", "ragging", "grievance", "career",
    "alumni", "research", "project", "women", "empowerment", "facilities", "lab",
    "labs", "class", "timetable", "mission", "vision", "accreditation", "mess",
    "chatbot", "name"
}

def is_college_related(user_input):
    words = set(user_input.split())
    return len(words.intersection(college_keywords)) > 0

def get_response(user_input):
    user_input_clean = clean_text(user_input)

    if not user_input_clean:
        return "Please type a question."

    if user_input_clean in ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]:
        return "Hello! How can I help you with college information?"

    if user_input_clean in ["thanks", "thank you"]:
        return "You are welcome."

    if user_input_clean in ["bye", "exit", "quit"]:
        return "Goodbye! Have a nice day."

    # Exact match
    for i, q in enumerate(cleaned_questions):
        if user_input_clean == q:
            return answers[i]

    # Partial match
    for i, q in enumerate(cleaned_questions):
        if user_input_clean in q or q in user_input_clean:
            return answers[i]

    # Reject clearly unrelated questions
    if not is_college_related(user_input_clean):
        return "Sorry, I can only answer questions related to the college."

    # Similarity match
    user_vector = vectorizer.transform([user_input_clean])
    similarity = cosine_similarity(user_vector, X)

    idx = similarity.argmax()
    score = similarity[0][idx]

    if score >= 0.55:
        return answers[idx]

    return "Sorry, I could not find the exact answer. Please ask about admissions, fees, exams, hostel, facilities, placements, certificates, or other college-related topics."

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
    user_input = request.form.get("msg", "")
    response = get_response(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)