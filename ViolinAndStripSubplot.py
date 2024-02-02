'''
 iterates through each feature in the dataset and
 creates a figure with two subplots:
 a strip plot and a violin plot.
 These plots show the distribution of data across different classes,
 helping to visualize how each feature varies with the class labels.
 The resulting figures are saved as image files for further analysis and interpretation.
'''

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('Data/Emotion_features.csv')
feature = data.loc[:, 'tempo':]
target = data['label']
targetName = data['class']
featureName = list(feature)

for name in featureName:
    plt.figure(figsize=(12, 12))
    
    plt.subplot(2,1,1)
    sns.stripplot(x='class', y=name, data=data, jitter=True)
    plt.title('Strip Plot for ' + name)
    
    plt.subplot(2,1,2)
    sns.violinplot(x='class', y=name, data=data)
    plt.title('Violin Plot for ' + name)
    
    plt.tight_layout()
    plt.savefig('Plots\\Violin and Strip Subplot\\' + name)
    plt.show()
    plt.clf()