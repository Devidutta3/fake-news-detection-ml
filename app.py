from flask import Flask, request, render_template_string
import pickle
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords (first run only)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load trained model & vectorizer
model = pickle.load(open("model/fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("model/tfidf_vectorizer.pkl", "rb"))

app = Flask(__name__)

# Text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# Simple HTML
html = """
<!DOCTYPE html>
<html>
<head>
    <title>Fake News Detection</title>
</head>
<body>
    <h2>📰 Fake News Detection System</h2>

    <form method="post">
        <textarea name="news" rows="10" cols="90"
        placeholder="Paste full news article here (minimum 20 words)..." required></textarea><br><br>
        <button type="submit">Check News</button>
    </form>

    {% if result %}
        <h3>{{ result }}</h3>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    if request.method == "POST":
        news = request.form["news"]
        cleaned = clean_text(news)

        # Length validation (VERY IMPORTANT)
        if len(cleaned.split()) < 20:
            result = "⚠️ Please enter a full news article (too short)"
        else:
            vectorized = vectorizer.transform([cleaned])
            prediction = model.predict(vectorized)[0]

            if prediction.strip().upper() == "REAL":
                result = "✅ REAL NEWS"
            else:
                result = "❌ FAKE NEWS"

    return render_template_string(html, result=result)

if __name__ == "__main__":
    app.run(debug=True)
