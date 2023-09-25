import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Load the data from 'Emotion_features.csv' (provide the correct path)
data = pd.read_csv('Emotion_features.csv')

# Select features from 'tempo' onwards
feature = data.loc[:, 'tempo':]
featureName = list(feature)

# Normalize the features
for name in featureName:
    feature[name] = (feature[name] - feature[name].min()) / (feature[name].max() - feature[name].min())

plt.style.use('ggplot')

# Convert the DataFrame to a NumPy array
array = np.array(data)

# Extract features and labels
features = feature.values
labels = data['class']  # Adjust the label column name as needed

test_size = 0.20
random_seed = 5

train_d, test_d, train_l, test_l = train_test_split(features, labels, test_size=test_size, random_state=random_seed)

result = []
xlabel = [i for i in range(1, 11)]
for neighbors in range(1, 11):
    kNN = KNeighborsClassifier(n_neighbors=neighbors)
    kNN.fit(train_d, train_l)
    prediction = kNN.predict(test_d)
    result.append(accuracy_score(prediction, test_l) * 100)

plt.figure(figsize=(10, 10))
plt.xlabel('kNN Neighbors for k=1,2...10')  # Adjust the label
plt.ylabel('Accuracy Score')
plt.title('kNN Classifier Results')
plt.ylim(0, 100)
plt.xlim(0, 10)
plt.plot(xlabel, result)
plt.savefig('1-fold 10NN Result.png')
plt.show()

# Accept 4 feature values as input for predicting emotions
new_audio_features = np.array([[8.27174432, 42.23890769, 1784.125323, -0.669535765]]).reshape(1, -1)

# Predict emotions for the new audio features using the trained k-NN model
predicted_emotion = kNN.predict(new_audio_features)
print("Predicted Emotion for New Audio:", predicted_emotion)
