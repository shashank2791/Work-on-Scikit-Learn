from sklearn.datasets import load_iris
import pandas as pd

path = load_iris()['filename']

df = pd.read_csv(path)
print(df.describe())