from flask import Flask, render_template, request
import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import re

MODEL_FILE = "model.pkl"
VECTORIZER_FILE = "vectorizer.pkl"
DATA_FILE = "Reddit_Data.csv"

app = Flask(__name__)

# 🔹 Clean text function
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower().strip()

# 🔹 Train & save model if not exists
def train_and_save_model():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError("❌ Dataset not found!")

    df = pd.read_csv(DATA_FILE).dropna(subset=['clean_comment'])
    df['clean_comment'] = df['clean_comment'].astype(str).apply(clean_text)
    df = df[df['clean_comment'].str.strip() != ""]

    X = df['clean_comment']
    y = df['category']

    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
    model = LinearSVC()
    model.fit(X_train, y_train)

    pickle.dump(model, open(MODEL_FILE, "wb"))
    pickle.dump(vectorizer, open(VECTORIZER_FILE, "wb"))
    print("✅ Model trained and saved!")
    return model, vectorizer

# 🔹 Load model & vectorizer
def load_model_and_vectorizer():
    try:
        model = pickle.load(open(MODEL_FILE, "rb"))
        vectorizer = pickle.load(open(VECTORIZER_FILE, "rb"))
        print("✅ Model loaded successfully.")
    except:
        model, vectorizer = train_and_save_model()
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        text = request.form["user_input"]
        cleaned_text = clean_text(text)
        transformed = vectorizer.transform([cleaned_text])
        pred = model.predict(transformed)[0]

        # ✅ Get confidence
        try:
            prob = abs(model.decision_function(transformed).max()) / 5
            prob = min(prob, 1)
        except:
            prob = 0.7

        # ✅ Final Scoring Logic
        if pred == 1:  # Positive
            score = int(60 + prob * 40)   # 60–100
            sentiment = "positive"
            emoji = "😊"
        elif pred == -1:  # Negative
            score = int(1 + prob * 33)    # 1–34
            sentiment = "negative"
            emoji = "😡"
        else:  # Neutral
            score = int(35 + prob * 24)   # 35–59
            sentiment = "neutral"
            emoji = "😐"

        return render_template("result.html",
                               text=text,
                               sentiment=sentiment,
                               score=score,
                               emoji=emoji)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
