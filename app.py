from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# ----------------------------
# FILE PATH
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "college_data.csv")

# ----------------------------
# LOAD DATA
# ----------------------------
try:
    data = pd.read_csv(CSV_PATH, encoding="utf-8", error_bad_lines=False)
except Exception:
    data = pd.read_csv(CSV_PATH, encoding="utf-8")

# Clean column names
data.columns = [col.strip().lower() for col in data.columns]

# Fix columns if needed
if "question" not in data.columns or "answer" not in data.columns:
    if len(data.columns) >= 3:
        data.columns = ["category", "question", "answer"]
    elif len(data.columns) == 2:
        data.columns = ["question", "answer"]
    else:
        raise ValueError("CSV format is incorrect. Expected question and answer columns.")

# Remove empty rows
data = data.dropna(subset=["question", "answer"])

questions = data["question"].astype(str).str.strip().tolist()
answers = data["answer"].astype(str).str.strip().tolist()

# ----------------------------
# TEXT CLEANING
# ----------------------------
def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_text(text):
    text = clean_text(text)

    replacements = {
        "admission procedure": "admission process",
        "procedure for admission": "admission process",
        "how do i join": "how can i apply for admission",
        "how to join": "how can i apply for admission",
        "joining process": "admission process",
        "cost": "fees",
        "price": "fees",
        "tuition": "fees",
        "exam schedule": "exams",
        "test schedule": "exams",
        "marksheet": "results",
        "accommodation": "hostel",
        "teacher": "faculty",
        "teachers": "faculty",
        "staff": "faculty",
        "tc": "transfer certificate",
        "bonafide": "bonafide certificate",
        "id": "id card",
        "attendance shortage": "low attendance",
        "job support": "placements",
        "placement job": "placements",
        "bus pass": "transport",
        "canteen": "cafeteria"
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text

cleaned_questions = [normalize_text(q) for q in questions]

# ----------------------------
# VECTORIZERS
# ----------------------------
word_vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2)
)
word_matrix = word_vectorizer.fit_transform(cleaned_questions)

char_vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5)
)
char_matrix = char_vectorizer.fit_transform(cleaned_questions)

# ----------------------------
# DOMAIN CHECK
# ----------------------------
college_keywords = {
    "admission", "admissions", "apply", "course", "courses", "fee", "fees",
    "exam", "exams", "hallticket", "hall", "ticket", "attendance", "hostel",
    "library", "placement", "placements", "syllabus", "faculty", "department",
    "college", "university", "campus", "transport", "scholarship", "results",
    "revaluation", "certificate", "bonafide", "transfer", "tc", "id", "office",
    "contact", "sports", "ncc", "nss", "ragging", "grievance", "career",
    "alumni", "research", "project", "women", "empowerment", "facilities",
    "facility", "lab", "labs", "class", "timetable", "mission", "vision",
    "accreditation", "mess", "history", "location", "timings", "leave",
    "counseling", "complaint", "internship", "cafeteria", "gym",
    "auditorium", "wifi", "principal", "study", "scholarships",
    "semester", "supplementary", "internal", "practical"
}

def is_college_related(user_input_clean):
    words = set(user_input_clean.split())
    return len(words.intersection(college_keywords)) > 0

# ----------------------------
# TOKEN OVERLAP
# ----------------------------
def token_overlap_score(a, b):
    a_words = set(a.split())
    b_words = set(b.split())

    if not a_words or not b_words:
        return 0.0

    return len(a_words.intersection(b_words)) / max(1, len(a_words))

# ----------------------------
# CONVERSATIONAL ANSWER LAYER
# ----------------------------
def make_conversational(answer):
    answer = answer.strip()

    if not answer:
        return "Sorry, I could not find the exact answer."

    lower_answer = answer.lower()

    if lower_answer.startswith("yes"):
        if len(answer.split()) < 10:
            return answer + " Students can contact the college office for more details if needed."
        return answer

    if lower_answer.startswith("no"):
        return answer + " Students are advised to follow official college guidelines for clarification."

    if "fee" in lower_answer:
        return answer + " For the latest fee details, students should verify with the college office or official notifications."

    if "exam" in lower_answer or "result" in lower_answer or "hall ticket" in lower_answer:
        return answer + " Students should regularly check official notices and the examination branch for updates."

    if "admission" in lower_answer:
        return answer + " Students should also keep all required documents ready and follow the official admission schedule."

    if "hostel" in lower_answer:
        return answer + " Hostel allotment usually depends on availability and hostel rules."

    if len(answer.split()) < 12:
        return answer + " Students can contact the college office for more information."

    return answer

# ----------------------------
# RESPONSE LOGIC
# ----------------------------
def get_response(user_input):
    user_input_clean = normalize_text(user_input)

    if not user_input_clean:
        return "Please type a question."

    greetings = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening"}
    thanks_words = {"thanks", "thank you"}
    bye_words = {"bye", "exit", "quit"}

    # Strict greeting check
    if user_input_clean in greetings:
        return "Hello! How can I help you with college information?"

    if user_input_clean in thanks_words:
        return "You are welcome."

    if user_input_clean in bye_words:
        return "Goodbye! Have a nice day."

    # Exact match
    for i, q in enumerate(cleaned_questions):
        if user_input_clean == q:
            return make_conversational(answers[i])

    # Reject clearly unrelated questions BEFORE soft matching
    if not is_college_related(user_input_clean):
        return "Sorry, I can only answer questions related to the college."

    # Token-based partial match instead of substring match
    best_partial_idx = -1
    best_partial_score = 0.0

    for i, q in enumerate(cleaned_questions):
        score = token_overlap_score(user_input_clean, q)
        if score > best_partial_score:
            best_partial_score = score
            best_partial_idx = i

    if best_partial_score >= 0.75:
        return make_conversational(answers[best_partial_idx])

    # Word similarity
    user_word_vec = word_vectorizer.transform([user_input_clean])
    word_scores = cosine_similarity(user_word_vec, word_matrix)[0]

    # Character similarity
    user_char_vec = char_vectorizer.transform([user_input_clean])
    char_scores = cosine_similarity(user_char_vec, char_matrix)[0]

    # Combined score
    combined_scores = (0.75 * word_scores) + (0.25 * char_scores)

    best_idx = combined_scores.argmax()
    best_score = combined_scores[best_idx]

    print("User Query:", user_input)
    print("Matched Question:", questions[best_idx])
    print("Combined Score:", best_score)
    print("Best Partial Score:", best_partial_score)

    if best_score >= 0.36:
        return make_conversational(answers[best_idx])

    return "Sorry, I could not find the exact answer. Please ask about admissions, fees, exams, hostel, facilities, placements, certificates, transport, or other college-related topics."

# ----------------------------
# ROUTES
# ----------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
    user_input = request.form.get("msg", "").strip()
    response = get_response(user_input)
    return jsonify({"response": response})

# ----------------------------
# RUN
# ----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)