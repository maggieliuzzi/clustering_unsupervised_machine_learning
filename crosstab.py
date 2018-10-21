import pandas as pd

df = pd.DataFrame({'labels': labels, 'species': species})
print(df)

ct = pd.crosstab(df['labels'], df['species'])
print(ct)
