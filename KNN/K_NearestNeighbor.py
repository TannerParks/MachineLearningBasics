import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data")
# print(data.head())

le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))  # transforms all the data to numerical values
maint = le.fit_transform(list(data["maint"]))  # fit_transform takes a list (our columns) and returns an array
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

predict = "class"

X = list(zip(buying, maint, door, persons, lug_boot, safety))  # features
y = list(cls)  # labels

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

# print(x_train, y_test)

model = KNeighborsClassifier(n_neighbors=9)  # parameter is the number of neighbors (should be odd so there's a winner)

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(f"Accuracy is: {acc*100}%")  # prints the accuracy

predicted = model.predict(x_test)   # predicts classification from 0-3 which is changed to words with the names list
names = ["unacc", "acc", "good", "vgood"] # unacceptable = 0, acceptable = 1, good = 2, very good = 3
# print(predicted)   # prints out classifications in integers

for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])
    # we can use Kneighbors to look at the neighbors of each point in our data
    n = model.kneighbors([x_test[x]], 9, return_distance=True)  # 9 = # of neighbors
    print("N: ", n)
