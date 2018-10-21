# Uses iris dataset
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import os

home_path = os.path.dirname(__file__)
train_path = home_path+"TrainingSet.csv"
trainingData = pd.read_csv(train_path)
test_path = home_path+"TestingSet.csv"
testingData = pd.read_csv(test_path)

model = KMeans(n_clusters=3)

model.fit(trainingData)
KMeans(algorithm='auto')

print(trainingData)
labels = model.predict(trainingData)

print(labels)
print(testingData)

new_labels = model.predict(testingData)
print(new_labels)

# Scatter Plot
xs = trainingData[:,0] # column 0
ys = trainingData[:,2] # column 2

plt.scatter(xs, ys, c=labels) # colour by cluster label

plt.show()

