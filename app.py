
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load your trained model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""

    if request.method == "POST":
        message = request.form.get("message", "")

        if message.strip():
            text = vectorizer.transform([message])
            prediction = model.predict(text)[0]

            if prediction == 1:
                result = "SPAM SMS"
            else:
                result = "NORMAL SMS"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run()
