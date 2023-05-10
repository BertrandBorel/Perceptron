# Perceptron class

import numpy as np


class Perceptron:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        # Initialisation des poids et du biais à zéro
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        
        # Entraînement du modèle
        for i in range(self.num_iterations):
            # Calcul de la prédiction
            y_hat = np.dot(X, self.weights) + self.bias
            
            # Calcul de l'erreur
            error = y - y_hat
            
            # Mise à jour des poids et du biais
            self.weights += self.learning_rate * np.dot(X.T, error)
            self.bias += self.learning_rate * np.sum(error)
            
    def predict(self, X):
        # Calcul de la prédiction
        y_hat = np.dot(X, self.weights) + self.bias
        
        # Application de la fonction d'activation (ici, la fonction seuil)
        y_pred = np.where(y_hat > 0, 1, 0)
        
        return y_pred