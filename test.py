import tensorflow
import keras
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.utils import shuffle
import sklearn
import matplotlib.pyplot as pyplot
from matplotlib import style
import pickle

data = pd.read_csv("student-mat.csv", sep=";")
#print(data.head())
data = data[["G1","G2","G3","studytime","failures","absences"]]

predict = "G3"
x = np.array(data.drop(["G3"],1))


y = np.array(data[predict])

#x_train , x_test, y_train ,y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.1)
best = 0
for _ in range(30):
    x_train , x_test, y_train ,y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.1) # Split arrays or matrices into random train and test subsets,split 10% of data into test samples,

    #train the data

    linear = linear_model.LinearRegression()
    linear.fit(x_train,y_train) # only trained by train data not test data
    acc = linear.score(x_test,y_test)
    print(acc)
    if acc>best:
        best = acc
        with open("studentmodel.pickle","wb") as f:
            pickle.dump(linear, f)
print("best:,", best)

pickle_in = open("studentmodel.pickle","rb")
linear = pickle.load(pickle_in)

print("Coefficient: ", linear.coef_)
print("Intercept: " , linear.intercept_)
predictions = linear.predict(x_test)
for i in range(len(predictions)):
    print(predictions[i],x_test[i], y_test[i])

p="absences"
style.use("ggplot")
pyplot.scatter(data[p],data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final grade")
pyplot.show()