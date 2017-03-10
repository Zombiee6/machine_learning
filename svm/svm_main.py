#!/usr/bin/python2

from accuarcy.calculate import calculate_accuracy
from classify_svm import classify
from terrain_data.class_vis import prettyPicture, output_image
from terrain_data.prep_terrain_data import makeTerrainData

features_train, labels_train, features_test, labels_test = makeTerrainData()

clf = classify(features_train, labels_train)

predict = clf.predict(features_test)

print calculate_accuracy(labels_test, predict)

prettyPicture(clf, features_test, labels_test)
output_image("test.png", "png", open("test.png", "rb").read())
