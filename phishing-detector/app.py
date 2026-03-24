from flask import Flask, render_template, request
import pickle
import re

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def contains_url(text):
    return bool(re.search(r"https?://|www\.", text))

def suspicious_keywords(text):
    keywords = ["urgent", "verify", "bank", "password", "click", "win", "reward"]
    return [word for word in keywords if word in text.lower()]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    email = request.form['email']

    data = vectorizer.transform([email])
    prediction = model.predict(data)[0]
    prob = model.predict_proba(data)[0][1]

    url_flag = contains_url(email)
    keywords = suspicious_keywords(email)

    if url_flag:
        result = "Suspicious: Contains URL"
    elif prediction == 1:
        result = f"Phishing ({prob*100:.2f}% confidence)"
    else:
        result = f"Safe ({(1-prob)*100:.2f}% confidence)"

    # Logging (fixed)
    with open("logs.txt", "a", encoding="utf-8") as f:
        f.write(email + " -> " + result + "\n")

    return render_template(
        'index.html',
        result=result,
        keywords=keywords,
        url_flag=url_flag
    )

if __name__ == "__main__":
    app.run(debug=True)