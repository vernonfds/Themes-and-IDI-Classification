import pandas as pd
import numpy as np
import re
import nltk
import joblib
import json
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import sys

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def clean_text(text):
    if pd.isnull(text):  # Handle NaN values
        return ""

    # Convert to lowercase
    text = str(text).lower()

    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Join the tokens back into a single string
    cleaned_text = ' '.join(tokens)

    return cleaned_text

# Load the saved models
classifiers = []
for i in range(10):  # Assuming you have 10 clusters as in the original code
    classifier = joblib.load(f'classifier_model_cluster{i}.joblib')
    classifiers.append(classifier)

# Load the vectorizer
vectorizer = joblib.load('tfidf_vectorizer.joblib')

# Load the fitted KMeans model
kmeans = joblib.load('kmeans_model.joblib')  # Make sure to save the KMeans model during training

# Define a function to classify a single message
def classify_single_message(message):
        
    cleaned_message = clean_text(message)
    # Check if the cleaned message contains 8 or more numbers
    if re.search(r"\b\d{19}\b", cleaned_message):
        return "Not Classified", 0.0
    message_features = vectorizer.transform([cleaned_message])

    # Perform clustering to identify patterns
    num_clusters = 10
    cluster = kmeans.predict(message_features)[0]
    classifier = classifiers[cluster]
    prediction = classifier.predict(message_features)[0]
    confidence_score = np.max(classifier.decision_function(message_features))
    return prediction, confidence_score

def classify_from_json(json_input):
    input_dict = json.loads(json_input)

    # Check if the message key exists in the input
    if "Brief Description of Feedback" not in input_dict:
        raise ValueError("Error: 'Brief Description of Feedback' key not found in the JSON input.")

    message = input_dict["Brief Description of Feedback"]
    prediction, confidence_score = classify_single_message(message)
    return {"Prediction": prediction, "Confidence Score": confidence_score}

if __name__ == "__main__":
    if len(sys.argv) > 1:
        json_input = sys.argv[1]
        try:
            result = classify_from_json(json_input)
            print(json.dumps(result, indent=4))
        except ValueError as e:
            print(str(e))
    else:
        print("Error: Invalid JSON input. Please provide the input in JSON format.")