#!/usr/bin/python2


from accuarcy.calculate import calculate_accuracy
from classify_nb import classify
from terrain_data.class_vis import prettyPicture, output_image
from terrain_data.prep_terrain_data import makeTerrainData

features_train, labels_train, features_test, labels_test = makeTerrainData()

### the training data (features_train, labels_train) have both "fast" and "slow" points mixed
### in together--separate them so we can give them different colors in the scatterplot,
### and visually identify them
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii] == 0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii] == 0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii] == 1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii] == 1]

clf = classify(features_train, labels_train)

predict = clf.predict(features_test)
print calculate_accuracy(labels_test, predict)

prettyPicture(clf, features_test, labels_test)
output_image("test.png", "png", open("test.png", "rb").read())
