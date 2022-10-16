import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics

cancer = datasets.load_breast_cancer()
# print(f"Features: {cancer.feature_names}") # x axis stuff (factors attributing to the classification)
# print(f"Targets: {cancer.target_names}")  # y axis (what it classifies as)

X = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

# print(x_train[:5], y_train[:5]) # print first 5 features and targets

classes = ["malignant", "benign"]   # classes we get by using target as an index: classes[0] = malignant

clf = svm.SVC(kernel="linear", C=2) # classifier = support vector classification # C is the soft margin
# add parameters (like kernel) to SVC to get a higher accuracy
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test) # predict values for our test data
acc = metrics.accuracy_score(y_test, y_pred)    # test against our correct values (y_test = actual y_pred = prediction)

print(acc)  # I'm pretending this isn't high by default so I can use a kernal parameter SVC()

for x in range(len(y_pred)):
    print(f"Predicted: {classes[y_pred[x]]}\t\t\tActual: {classes[y_test[x]]}")
