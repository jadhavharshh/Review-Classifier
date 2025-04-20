from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import re
from io import BytesIO

# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import base64

STOPWORDS = set(stopwords.words("english"))

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})

@app.route("/test", methods=["GET"])
def test():
    return "Test request received successfully. Service is running."


@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("landing.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Select the predictor to be loaded from Models folder
    predictor = pickle.load(open(r"Models/model_xgb.pkl", "rb"))
    scaler = pickle.load(open(r"Models/scaler.pkl", "rb"))
    cv = pickle.load(open(r"Models/countVectorizer.pkl", "rb"))
    try:
        # Check if the request contains a file (for bulk prediction) or text input
        if "file" in request.files:
            # Bulk prediction from CSV file
            file = request.files["file"]
            data = pd.read_csv(file)

            predictions, graph = bulk_prediction(predictor, scaler, cv, data)

            response = send_file(
                predictions,
                mimetype="text/csv",
                as_attachment=True,
                download_name="Predictions.csv",
            )

            response.headers["X-Graph-Exists"] = "true"

            response.headers["X-Graph-Data"] = base64.b64encode(
                graph.getbuffer()
            ).decode("ascii")

            return response

        elif "text" in request.json:
            # Single string prediction
            text_input = request.json["text"]
            predicted_sentiment = single_prediction(predictor, scaler, cv, text_input)

            return jsonify({"prediction": predicted_sentiment})

    except Exception as e:
        return jsonify({"error": str(e)})


def single_prediction(predictor, scaler, cv, text_input):
    # First check for presence of negative words
    negative_words = [
        "bad", "terrible", "awful", "horrible", "poor", "waste", 
        "disappointed", "useless", "broken", "issues", "problem", 
        "difficult", "slow", "expensive", "overpriced", "regret",
        "not work", "doesn't work", "not worth", "frustrating",
        "annoying", "hate", "dislike", "worst", "junk", "failure",
        "defective", "faulty", "unreliable", "cheap", "sucks",
        "refund", "return", "complaint", "unhappy", "dissatisfied"
    ]
    
    # Convert to lowercase for case-insensitive matching
    text_lower = text_input.lower()
    
    # Check if any negative word is in the input text
    for word in negative_words:
        if word in text_lower:
            return "Negative"  # Immediately return negative if keyword found
    
    # If no negative keywords found, proceed with ML classification
    corpus = []
    stemmer = PorterStemmer()
    review = re.sub("[^a-zA-Z]", " ", text_input)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if not word in STOPWORDS]
    review = " ".join(review)
    corpus.append(review)
    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    
    # Get probability scores
    probabilities = predictor.predict_proba(X_prediction_scl)[0]
    
    # Use a lower threshold for negative class (index 0)
    # This helps counter the positive bias in the training data
    if probabilities[0] > 0.2:
        return "Negative"
    else:
        return "Positive"


def bulk_prediction(predictor, scaler, cv, data):
    negative_words = [
        "bad", "terrible", "awful", "horrible", "poor", "waste", 
        "disappointed", "useless", "broken", "issues", "problem", 
        "difficult", "slow", "expensive", "overpriced", "regret",
        "not work", "doesn't work", "not worth", "frustrating",
        "annoying", "hate", "dislike", "worst", "junk", "failure"
    ]
    
    corpus = []
    stemmer = PorterStemmer()
    predictions = []
    
    for i in range(0, data.shape[0]):
        text = data.iloc[i]["Sentence"]
        text_lower = text.lower()
        
        # First check for negative keywords
        negative_found = False
        for word in negative_words:
            if word in text_lower:
                predictions.append("Negative")
                negative_found = True
                break
                
        if not negative_found:
            # Process with ML model if no keywords found
            review = re.sub("[^a-zA-Z]", " ", text)
            review = review.lower().split()
            review = [stemmer.stem(word) for word in review if not word in STOPWORDS]
            review = " ".join(review)
            corpus.append(review)
            
            # Process through model after corpus is built
            X_prediction = cv.transform([review]).toarray()
            X_prediction_scl = scaler.transform(X_prediction)
            probs = predictor.predict_proba(X_prediction_scl)[0]
            
            if probs[0] > 0.2:
                predictions.append("Negative")
            else:
                predictions.append("Positive")
    
    data["Predicted sentiment"] = predictions
    predictions_csv = BytesIO()
    data.to_csv(predictions_csv, index=False)
    predictions_csv.seek(0)
    
    graph = get_distribution_graph(data)
    
    return predictions_csv, graph

def get_distribution_graph(data):
    fig = plt.figure(figsize=(5, 5))
    colors = ("green", "red")
    wp = {"linewidth": 1, "edgecolor": "black"}
    tags = data["Predicted sentiment"].value_counts()
    explode = (0.01, 0.01)

    tags.plot(
        kind="pie",
        autopct="%1.1f%%",
        shadow=True,
        colors=colors,
        startangle=90,
        wedgeprops=wp,
        explode=explode,
        title="Sentiment Distribution",
        xlabel="",
        ylabel="",
    )

    graph = BytesIO()
    plt.savefig(graph, format="png")
    plt.close()

    return graph


def sentiment_mapping(x):
    if x == 1:
        return "Positive"
    else:
        return "Negative"


if __name__ == "__main__":
    app.run(port=5000, debug=True)
