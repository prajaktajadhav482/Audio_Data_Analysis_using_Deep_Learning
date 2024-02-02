'''
performs kNN classification for audio features related to emotions.
It normalizes the features,
evaluates the model's performance for different k values,
creates plots to visualize the results,
and saves these plots as image files.
This process is repeated for all the features in csv file,
allowing you to assess the performance of kNN classification for each feature.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv('Data/Emotion_features.csv')
feature = data.loc[:, 'tempo':]
featureName = list(feature)
color = ['red' if l==1 else 'green' if l==2 else 'blue' if l==3 else 'orange' for l in data['label']]

plt.style.use('ggplot')

array = np.array(data)

for iterator in range(4, len(array)):
    features = array[:, iterator]
    
    # Normalization
    features = np.abs(features-features.mean())/(features.max()-features.min())
    
    labels = data.loc[:, 'class'].dropna()
    test_size = 0.20
    random_seed = 7
    
    train_d, test_d, train_l, test_l = train_test_split(features, labels, test_size=test_size, random_state=random_seed)
    
    train_l = train_l.ravel()
    test_l = test_l.ravel()
    
    result = []
    xlabel = [i for i in range(1, 11)]
    for neighbors in range(1, 11):
        kNN = KNeighborsClassifier(n_neighbors=neighbors)
        kNN.fit(train_d.reshape(-1, 1), train_l)
        prediction = kNN.predict(test_d.reshape(-1, 1))
        result.append(accuracy_score(prediction, test_l)*100)
    
    plt.figure(figsize=(10, 10))
    plt.xlabel('kNN Neighbors for k=1,2...10')
    plt.ylabel('Accuracy Score')
    plt.title('kNN Classifier Result for ' + featureName[iterator])
    plt.ylim(0, 100)
    plt.xlim(0, xlabel[len(xlabel)-1]+1)
    plt.plot(xlabel, result)
    plt.savefig('Figure\\Individual\\Normalized\\' + featureName[iterator] + '.png')
    plt.show()
