import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")  # data is separated by semicolons in csv so sep needs to equal ";"

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
predict = "G3"  # aka the final grade

X = np.array(data.drop([predict], 1))   # Features
y = np.array(data[predict]) # Labels

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1) # 0.1 means 10% of available data is used

# this for loop was used to train the model until it reached 95%+ accuracy
"""best = 0    # best accuracy
for _ in range(300): # use a loop to get a high accuracy for our model
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)    # finds best fit line
    acc = linear.score(x_test, y_test)  # acc = accuracy
    print(acc)  # training will be different every time so this number will fluctuate a little

    if acc > best:  # update pickle file when accuracy is more than the best so far
        best = acc # updates the best variable to the new best
        with open("studentmodel.pickle", "wb") as f:    # makes a pickle file to save our model
            pickle.dump(linear, f)
print("best is: " + str(best))"""

pickle_in = open("studentmodel.pickle", "rb")   # doing this allows us to comment out the above code after running it
linear = pickle.load(pickle_in) # loads our trained model (96%) into the variable called linear

print("Coefficient:\n", linear.coef_)   # 5 coefficients for multidimensional space y=mx+nz+pq+wt+ka+b
print("Intercept:\n", linear.intercept_)

predictions = linear.predict(x_test)    # generates a grade prediction based on the data (G1, studytime, etc)

for x in range(len(predictions)):
    print(f"Prediction: {predictions[x]} ", f"Data:  {x_test[x]} ", f"Actual Grade: {y_test[x]}")

p = "G1"    # change this to see the correlation of other data on the final grade (G3)
style.use("ggplot")
pyplot.scatter(data[p], data["G3"]) # x axis is data inputted to p, y is grade 3 aka final grade
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()
