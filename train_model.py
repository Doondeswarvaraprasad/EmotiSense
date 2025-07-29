import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("Reddit_Data.csv")

# ✅ Drop NaN values
df = df.dropna(subset=['clean_comment'])

# ✅ Ensure all comments are strings
df['clean_comment'] = df['clean_comment'].astype(str)

# ✅ Clean text function
def clean_text(text):
    text = re.sub(r"http\S+", "", text)        # Remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)    # Remove special characters/numbers
    text = text.lower().strip()
    return text

df['clean_comment'] = df['clean_comment'].apply(clean_text)

# ✅ Remove empty strings after cleaning
df = df[df['clean_comment'].str.strip() != ""]

# Features & Labels
X = df['clean_comment']
y = df['category']

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Train SVM Model
model = LinearSVC()
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
print("✅ Model Accuracy:", accuracy_score(y_test, y_pred))

# Save Model & Vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
print("✅ Model and vectorizer saved successfully!")
