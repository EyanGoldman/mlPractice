import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Input training data
training_points = [[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]]
training_labels = [1, 1, 1, 2, 2, 2]
X = np.array(training_points)
Y = np.array(training_labels)

# Create Naive Bayes classifier
clf = GaussianNB()
clf.fit(X, Y)

# Classify test data with the classifier
test_points = [[1, 1], [2, 2], [3, 3], [4, 3]]
test_labels = [2, 2, 2, 2]
predicts = clf.predict(test_points)
print(predicts, accuracy_score(test_labels, predicts))