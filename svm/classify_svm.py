def classify(features_train, labels_train):
    from sklearn.svm import SVC
    # clf = SVC(kernel="linear")
    clf = SVC(1000, kernel="rbf")
    return clf.fit(features_train, labels_train)
