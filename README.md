import numpy as np
from sklearn.naive_bayes import GaussianNB

X = np.array([
    [25, 65],  
    [30, 70],  
    [28, 80],  
    [35, 75],  
])

y = np.array([1, 1, 0, 0])

clf = GaussianNB()
clf.fit(X, y)

new_point = np.array([[27, 72]])
prediction = clf.predict(new_point)
probabilities = clf.predict_proba(new_point)

print(f"Prediction for point (27, 72) is Class: {prediction[0]}")
print(f"Probabilities: {probabilities}")
