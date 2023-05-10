import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from flask import Flask, redirect, render_template, request, url_for

app = Flask(__name__)

# Initialize BERT model and tokenizer
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def analyze_sentiment(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    # Get the predicted sentiment label
    outputs = model(**inputs)
    sentiment_scores = torch.softmax(outputs.logits, dim=1).tolist()[0]
    sentiment_label = torch.argmax(outputs.logits).item()
    # Map the sentiment label to a sentiment string
    if sentiment_label == 0:
        sentiment = "negative"
    elif sentiment_label == 1:
        sentiment = "neutral"
    else:
        sentiment = "positive"
    return sentiment, sentiment_scores

@app.route("/", methods=("GET", "POST"))
def index():
    if request.method == "POST":
        text = request.form["mood"]
        sentiment, sentiment_scores = analyze_sentiment(text)
        sentiment_scores = [round(score, 3) for score in sentiment_scores]
        sentiment_scores_dict = dict(zip(text.split(), sentiment_scores))
        total_sentiment_score = round(sum(sentiment_scores), 3)
        return redirect(url_for("index", result=sentiment, scores=sentiment_scores_dict, total_score=total_sentiment_score))

    result = request.args.get("result")
    scores = request.args.get("scores")
    total_score = request.args.get("total_score")
    if scores:
        scores = eval(scores)
    return render_template("index.html", result=result, scores=scores, total_score=total_score)
