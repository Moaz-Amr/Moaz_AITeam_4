python
import numpy as np

class LinearRegression:
    def __init__(self, lr=0.01, iterations=1000):
        self.lr = lr
        self.iterations = iterations
        self.m = None
        self.b = None
    
    def fit(self, X, y):
        samples, features = X.shape
        self.m = np.zeros(features)
        self.b = 0
        
        for _ in range(self.iterations):
            predictions = np.dot(X, self.m) + self.b
            slope_error = (1/samples) * np.dot(X.T, (predictions - y))
            intercept_error = (1/samples) * np.sum(predictions - y)
            
            self.m = self.m - self.lr * slope_error
            self.b = self.b - self.lr * intercept_error
    
    def predict(self, X):
        return np.dot(X, self.m) + self.b