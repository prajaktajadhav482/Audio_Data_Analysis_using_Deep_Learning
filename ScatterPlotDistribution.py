'''
 processes audio feature data,
 creates scatter plots for each feature,
 assigns colors to different class labels,
 saves the plots as image files, and displays them for visualization.
 The result is a set of scatter plots showing the distribution of audio features across different classes.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Data/Emotion_features.csv')
features = data.loc[:, 'tempo':]
class_labels = data['label']

# Define a color map for class labels
color_map = {1: 'red', 2: 'green', 3: 'blue', 4: 'orange'}

plt.style.use('ggplot')

# Iterate over each feature
for feature_name in features.columns:
    plt.figure(figsize=(12, 12))
    plt.xlabel('Class')
    plt.ylabel(feature_name)
    plt.title(feature_name + ' Distribution')

    # Create a scatter plot for the current feature
    plt.scatter(class_labels, features[feature_name], c=[color_map[label] for label in class_labels])

    plt.savefig('Figure/ScatterPlot/' + feature_name + '.png')
    plt.show()
    plt.clf()
