def calculate_accuracy(true, predict):
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(true, predict)
    return acc
