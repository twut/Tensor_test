import numpy as np
import pandas as pd

from sklearn import linear_model
from sklearn.utils import shuffle
import sklearn

data = pd.read_csv("student-mat.csv", sep=";")
#print(data.head())
data = data[["G1","G2","G3","studytime","failures","absences"]]
print(data.shape)
predict = "G3"
x = np.array(data.drop(["G3"],0))
print(x.shape)