from NBTerrainData.prep_terrain_data import makeTerrainData
from classify import NBAccuracy

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl

def submitAccuracy():
    accuracy = NBAccuracy(features_train, labels_train, features_test, labels_test)
    return accuracy

features_train, labels_train, features_test, labels_test = makeTerrainData()
accuracy = submitAccuracy()
print accuracy

