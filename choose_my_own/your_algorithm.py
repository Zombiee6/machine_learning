#!/usr/bin/python

import matplotlib.pyplot as plt
from nltk import DecisionTreeClassifier

from terrain_data.prep_terrain_data import makeTerrainData
from terrain_data.class_vis import prettyPicture
from accuarcy.calculate import calculate_accuracy

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

#KNN
# clf = KNeighborsClassifier(n_neighbors=1)
# clf.fit(features_train, labels_train)

#Random Forest
# clf = RandomForestClassifier(n_estimators=100)
# clf = clf.fit(features_train, labels_train)

#AdaBoost
clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=200)
clf = clf.fit(features_train, labels_train)


predict = clf.predict(features_test)
print calculate_accuracy(labels_test, predict)


try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
