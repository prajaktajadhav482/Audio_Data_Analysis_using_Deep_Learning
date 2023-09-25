import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import csv
import random

# Load the emotion detection model and labels
model = load_model("detectormodel.h5")
label = np.load("emotionlabels.npy")

# Load the CSV file containing emotion-song mappings
emotion_songs = {}
with open("Data/songemotions.csv", "r") as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)  # Read and store the header row
    for row in reader:
        if len(row) != 2:
            print(f"Invalid row: {row}. Skipping.")
            continue

        song, emotion = row
        if emotion not in emotion_songs:
            emotion_songs[emotion] = []
        emotion_songs[emotion].append(song)

# Add "relax" labeled songs if they exist
if "relax" in emotion_songs:
    relax_songs = emotion_songs["relax"]
else:
    relax_songs = []

holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    lst = []

    _, frm = cap.read()

    frm = cv2.flip(frm, 1)

    res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

    if res.face_landmarks:
        for i in res.face_landmarks.landmark:
            lst.append(i.x - res.face_landmarks.landmark[1].x)
            lst.append(i.y - res.face_landmarks.landmark[1].y)

        if res.left_hand_landmarks:
            for i in res.left_hand_landmarks.landmark:
                lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
        else:
            for i in range(42):
                lst.append(0.0)

        if res.right_hand_landmarks:
            for i in res.right_hand_landmarks.landmark:
                lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
        else:
            for i in range(42):
                lst.append(0.0)

        lst = np.array(lst).reshape(1, -1)

        # Predict the detected emotion
        detected_emotion = label[np.argmax(model.predict(lst))]

        # Recommend a song based on the detected emotion
        #for model.h5 and labels.npy
        # if detected_emotion == "neutral":
        #     recommended_song = random.choice(relax_songs) if relax_songs else "No relax song found"
        # elif detected_emotion == "sad":
        #     # Suggest songs with label "angry" in addition to "sad"
        #     # possible_emotions = ["sad", "angry"]
        #     recommended_song = random.choice(emotion_songs[random.choice(possible_emotions)]) if any(emotion_songs[detected_emotion] for detected_emotion in possible_emotions) else "No matching song found"
        # elif detected_emotion == "surprised":
        #     # Suggest any random song without checking labels
        #     recommended_song = random.choice(list(emotion_songs.values())[0]) if emotion_songs else "No songs found"
        # elif detected_emotion in emotion_songs:
        #     songs_for_emotion = emotion_songs[detected_emotion]
        #     recommended_song = random.choice(songs_for_emotion)
        # else:
        #     recommended_song = "No song found for this emotion"

        #for detectormodel.h5 and emotionlabels.npy
        if "angry" in emotion_songs and "angry" == detected_emotion:
            recommended_song = random.choice(emotion_songs["angry"]) if "angry" in emotion_songs else "No matching song found"
        elif "happy" in emotion_songs and "happy" == detected_emotion:
            recommended_song = random.choice(emotion_songs["happy"]) if "happy" in emotion_songs else "No matching song found"
        elif "relax" in emotion_songs and "relax" == detected_emotion:
            recommended_song = random.choice(emotion_songs["relax"]) if "relax" in emotion_songs else "No matching song found"
        elif "sad" in emotion_songs and "sad" == detected_emotion:
            recommended_song = random.choice(emotion_songs["sad"]) if "sad" in emotion_songs else "No matching song found"
        elif detected_emotion == "surprised":
            # Suggest any random song without checking labels
            recommended_song = random.choice(list(emotion_songs.values())[0]) if emotion_songs else "No songs found"
        else:
            recommended_song = "No song found for this emotion"

        #print results
        print(f"Detected Emotion: {detected_emotion}")
        print(f"Recommended Song: {recommended_song}")

        cv2.putText(frm, f"Emotion: {detected_emotion}", (50, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)
        # cv2.putText(frm, f"Song: {recommended_song}", (50, 100), cv2.FONT_ITALIC, 1, (0, 0, 255), 2)

    cv2.imshow("window", frm)

    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        cap.release()
        break
