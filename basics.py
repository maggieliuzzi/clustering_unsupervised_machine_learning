# Uses iris dataset
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import os
import csv

home_path = os.path.dirname(__file__)
train_path = home_path+"TrainingSet.csv"
trainingData = pd.read_csv(train_path)
test_path = home_path+"TestingSet.csv"
testingData = pd.read_csv(test_path)

bedrooms_list = []
price_list = []
qualified_list = []
with open(train_path) as f:
    f.next()
    reader = csv.reader(f)
    for line in reader:
        bedrooms = line[9]
        price = line[15]
        qualified = line[16]
        qualified = qualified.replace("'","")
        if price is not '' and price is not '0' and bedrooms is not '':
            bedrooms_list.append(int(bedrooms))
            price_list.append(int(price))
            qualified_list.append(qualified)
    print(bedrooms_list)
    print(price_list)
    print(qualified_list)


# trainingData.plot()  # plots all columns against index
# trainingData.plot(kind='scatter',x='x',y='y') # scatter plot
# trainingData.plot(kind='density')  # estimate density function
# trainingData.plot(kind='hist')  # histogram


# Set the date column as the index of your DataFrame discoveries
trainingData = trainingData.set_index('PRICE')


# Scatter Plot
xs = price_list # column 2 [:,2]
ys = bedrooms_list # column 0 [:,0]

plot = plt.scatter(xs, ys) # c=labels colour by cluster label

plot.set_title('Price & Bedrooms', fontsize=16)

# Specify the x-axis label in your plot
plot.set_xlabel('Price', fontsize=13)

# Specify the y-axis label in your plot
plot.set_ylabel('Bedrooms', fontsize=13)

plot.style.use('ggplot')

plt.show()


'''
# Clustering 2D points

# points: trainingPoints
# new_points: testingPoints (array of points)
# labels: array of their clustered labels

# Import KMeans
from sklearn.cluster import KMeans

# Create a KMeans instance with 3 clusters: model
model = KMeans(n_clusters=3)

# Fit model to points
model.fit(points)

print(model.inertia_) # distance to centroid of cluster

# Determine the cluster labels of new_points: labels
labels = model.predict(new_points)

# Or use fit_predict to fit model and obtain cluster labels: labels
# labels = model.fit_predict(samples) # same as doing fit and then predict

# Print cluster labels of new_points
print(labels)

# Import pyplot
import matplotlib.pyplot as plt

# Assign the columns of new_points: xs and ys
xs = new_points[:,0]
ys = new_points[:,1]

# Make a scatter plot of xs and ys, using labels to define the colors
plt.scatter(xs, ys, c=labels, alpha=0.5) # to color points by their cluster label

# Assign the cluster centers: centroids
centroids = model.cluster_centers_

# Assign the columns of centroids: centroids_x, centroids_y
centroids_x = centroids[:,0]
centroids_y = centroids[:,1]

# Make a scatter plot of centroids_x and centroids_y
plt.scatter(centroids_x, centroids_y, marker='D', s=50) # using 'D' (a diamond) as a marker, size of markers: 50
plt.show()

'''



