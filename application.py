from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

application = Flask(__name__)

@application.route("/")
def index():
    return "Your Flask App Works! V1.0"

def load_model():
    with open('basic_classifier.pkl', 'rb') as fid:
        model = pickle.load(fid)
    with open('count_vectorizer.pkl', 'rb') as vd:
        vectorizer = pickle.load(vd)
    return model, vectorizer

if __name__ == "__main__":
    application.run(port=5000, debug=True)