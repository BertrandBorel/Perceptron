import numpy as np
from Perceptron_class import Perceptron



# jeu de données
X_train = np.array([[2, 4], [4, 2], [5, 5], [7, 6]])
y_train = np.array([0, 0, 1, 1])

# initiation d'un objet
perceptron = Perceptron(learning_rate=0.01, num_iterations=1000)

# entraînement du perceptron
perceptron.fit(X_train, y_train)

# jeu de test
X_test = np.array([[0, 1], [6, 5]])
# retourne une prédiction binaire pour chaque échantillon de test
y_pred = perceptron.predict(X_test)

# résultats de la prédiction
print(y_pred)