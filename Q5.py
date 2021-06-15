from sklearn.linear_model import LinearRegression 
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pandas as pd
boston = load_boston()
boston.data.shape, boston.target.shape
bos = pd.DataFrame(boston.data)
bos.columns = boston.feature_names
print(bos.head())
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2)
X1_train, X1_test, y1_train, y1_test = train_test_split(boston.data, boston.target, test_size=0.2)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
sklinreg = LinearRegression(normalize=True)
sklinreg.fit(X_train, y_train)
print("\nLinear Regrssion : \n")
print("Train:", sklinreg.score(X_train, y_train))
print("Test:", sklinreg.score(X_test, y_test))

from sklearn.linear_model import LogisticRegression as SKLR
X1_train, X1_test, y1_train, y1_test = train_test_split(boston.data, boston.target, test_size=0.2)

sk_logreg = SKLR()

sklinreg.fit(X1_train, y1_train)
print("\nLogistic Regrssion : \n")
# sk_logreg.score(X_test, y_test)
#Logistic regression will not work for boston dataset because 
# it is not a classification problem but a regression problem.  