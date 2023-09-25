import numpy as np
import pandas as pd
from python_speech_features import mfcc
import librosa
import os
import math
import pickle
import random
import operator
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the KNN model and genre labels
with open("knn_model.pkl", "rb") as f:
    knn_model = pickle.load(f)

with open("genre_labels.pkl", "rb") as labels_file:
    genre_labels = pickle.load(labels_file)


# Define a function to extract audio features from an MP3 file
def extract_audio_features(file_path):
    y, sr = librosa.load(file_path)
    mfcc_feat = mfcc(y, sr, winlen=0.020, appendEnergy=False)
    covariance = np.cov(np.transpose(mfcc_feat))
    mean_matrix = mfcc_feat.mean(0)
    return mean_matrix, covariance


# Route for the homepage
@app.route("/")
def home():
    return render_template("index.html")


# Route for processing the uploaded MP3 file and making a prediction
@app.route("/upload", methods=["POST"])
def upload_and_predict():
    try:
        # Check if a file was uploaded
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        # Check if the file has an allowed extension (e.g., .mp3)
        if file and file.filename.endswith(".mp3"):
            # Save the uploaded file to a temporary directory
            uploaded_file_path = os.path.join("temp", file.filename)
            file.save(uploaded_file_path)

            # Extract audio features from the uploaded file
            uploaded_features = extract_audio_features(uploaded_file_path)

            # Predict the genre based on the uploaded features
            predicted_genre = knn_model.predict([uploaded_features])[0]

            # Get the genre label for the prediction
            predicted_genre_label = genre_labels[predicted_genre]

            return jsonify({"genre": predicted_genre_label}), 200

        else:
            return jsonify({"error": "Invalid file format. Please upload an MP3 file."}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Create a temporary directory for uploaded files
    os.makedirs("temp", exist_ok=True)

    # Start the Flask app
    app.run(debug=True)
