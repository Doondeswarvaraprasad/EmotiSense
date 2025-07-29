🚀 EmotiSense – AI-Powered Sentiment Analysis Web Application
EmotiSense is an end-to-end machine learning web app that analyzes user text and outputs:

🎯 Sentiment Classification: Positive, Neutral, or Negative

📊 Confidence-based Score (1–100)

🎨 Dynamic UI with adaptive background and animated score meter

📌 Table of Contents
✨ Features

🛠️ Tech Stack

📂 Project Structure

⚙️ How It Works

💻 Local Setup

🌍 Deployment on Render

📈 Example Predictions

🔮 Future Enhancements

👤 Author

✨ Features
🤖 ML-powered sentiment scoring with numeric range

😐 Three Sentiment Classes: Negative (1–34), Neutral (35–59), Positive (60–100)

🎨 Dynamic UI with animated score meter & adaptive backgrounds

⚡ Flask backend with fast predictions

🌍 Easy Render deployment from GitHub

🛠️ Tech Stack
🔹 Component	🔹 Technology
Frontend	HTML5, CSS3 (custom animations, responsive)
Backend	Flask (Python)
ML Model	LinearSVC (Scikit-learn) + TF-IDF
Deployment	Render + Gunicorn

📂 Project Structure
php
Copy
Edit
sentiment_app/
│── app.py                # Flask main app
│── train_model.py        # ML model training script
│── Reddit_Data.csv       # Dataset
│── model.pkl             # Trained sentiment model
│── vectorizer.pkl        # Saved TF-IDF vectorizer
│── requirements.txt      # Dependencies
│── Procfile              # Render configuration
│
├── templates/
│   ├── index.html        # Input page
│   └── result.html       # Result display page
│
└── static/
    ├── style.css         # CSS & animations
    ├── logo.png          # Application logo
    ├── bg_positive.png   # Background for positive sentiment
    ├── bg_neutral.png    # Background for neutral sentiment
    ├── bg_negative.png   # Background for negative sentiment
    └── bg_default.png    # Default background
⚙️ How It Works
🧹 Input Preprocessing → Cleans and prepares text.

🔢 Feature Extraction → TF-IDF vectorization.

🧠 Prediction → LinearSVC classifies sentiment.

📈 Score Calculation → Generates a confidence-based score (1–100).

🎨 UI Rendering → Displays results with dynamic visuals.

💻 Local Setup
bash
Copy
Edit
# 1️⃣ Clone repository
git clone https://github.com/Doondeswarvaraprasad/EmotiSense.git
cd EmotiSense

# 2️⃣ (Optional) Create virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Train the model (if required)
python train_model.py

# 5️⃣ Run the Flask app
python app.py
🌐 Open in browser:
http://127.0.0.1:5000/

🌍 Deployment on Render
Push your code to GitHub.

Create a Web Service on Render.

Configure:

Build Command: pip install -r requirements.txt

Start Command: gunicorn app:app

🚀 Deploy and use your live URL.

📈 Example Predictions
📝 Input Text	🎯 Prediction	📊 Score
"This project is excellent"	Positive	82
"It’s okay, nothing special"	Neutral	51
"Worst experience ever"	Negative	18

🔮 Future Enhancements
📌 Use transformer-based models (BERT) for better accuracy

🌐 Add multilingual sentiment support

🐳 Provide Dockerized deployment

🔌 Offer REST API endpoints for integration

👤 Author
Tammina Doondeswara Prasad

🔗 LinkedIn: https://linkedin.com/in/tammina-doondeswar-31910b22b

🔗 GitHub: https://github.com/Doondeswarvaraprasad

📧 Email: tamminadoondeswar@gmail.com

