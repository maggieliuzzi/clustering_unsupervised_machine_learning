# Other scikit-learn pre-processing tools include Normalizer and MaxAbsScaler.
# While StandardScaler() standardizes features by removing the mean and scaling to unit variance,
# Normalizer() rescales each sample independently of the other.

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

# samples: trainingData

# normalizer = Normalizer()

scaler = StandardScaler()
scaler.fit(samples)
StandardScaler(copy=True, with_mean=True, with_std=True)
samples_scaled = scaler.transform(samples)

kmeans = KMeans(n_clusters=3)

pipeline = make_pipeline(scaler, kmeans)
pipeline.fit(samples) # Numpy array

labels = pipeline.predict(samples) # to get clustered labels # also Numpy array

df = pd.DataFrame({'labels': labels, 'species': species})
print(df)
# print(df.sort_values('labels'))

ct = pd.crosstab(df['labels'], df['species'])
print(ct)


