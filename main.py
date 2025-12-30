import numpy as np
from knn import KNNClassifier, load_xy_from_txt

X_train, y_train = load_xy_from_txt("data/IRIS.csv")
X_test,  y_test  = load_xy_from_txt("data/iris_test.csv")

knn = KNNClassifier(k=3).fit(X_train, y_train)
y_pred = knn.predict(X_test)

acc = np.mean(y_pred == y_test)
print("Accuracy:", acc)
