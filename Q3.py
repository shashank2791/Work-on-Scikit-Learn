from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd

path = load_iris()['filename']

df = pd.read_csv(path)

X = df.iloc[:, :-1].values # first 4 columns of the data set
y = df.iloc[:, 4].values # labels 

#Split arrays or matrices into random train and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
print("\n70% train data:")
print(X_train)
print(y_train)
print("\n30% test data:")
print(X_test)
print(y_test)