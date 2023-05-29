import numpy as np

# TODO: Fill out class
# TODO: Tests
# TODO: docstrings
# TODO: error/sanity checking

class LinearRegression:
    # Class variables
    # Instance variables
    
    def fit(self):
        pass
    
    def predict(self, X, w, b):
        return np.dot(X, w) + b
    
    def compute_cost(self, X, y, w, b, cost_fn=None):
        if not cost_fn:
            cost_fn = self.MSE
        return cost_fn(X, y, w, b)
    
    def MSE(self, X, y, w, b):        
        m = X.shape[0]
        err = self.predict(X, w, b) - y
        sq_err = np.power(err, 2)
        mse = np.sum(sq_err) / (2 * m)
        return mse
    
    def compute_gradient(self, X, y, w, b):
        pass
    
    def gradient_descent(self):
        pass
    