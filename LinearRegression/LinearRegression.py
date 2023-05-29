import numpy as np
from config import *

# TODO: Fill out class
# TODO: Tests
# TODO: docstrings
# TODO: error/sanity checking
# TODO: Regularization
# TODO: Normal Equation solution
# TODO: Over/under fit and common solutions, automatic

class LinearRegression:
    # Class variables

    def __init__(self):
        self.w = None
        self.b = None
    
    def fit(self, X, y, init_random=False):
        m, n = X.shape
        w, b = self.init_weights(n, init_random)
        self.w, self.b, _ = self.gradient_descent(X, y, w, b)
    
    def predict(self, X, w, b):
        return np.dot(X, w) + b
    
    def init_weights(self, n, random=False):
        if not random:
            w = np.zeros(n)
            b = 0
        else:
            # TODO: This branch
            pass
        return w, b
    
    def MSE(self, X, y, w, b):        
        m = X.shape[0]
        err = self.predict(X, w, b) - y
        sq_err = np.power(err, 2)
        mse = np.sum(sq_err) / (2 * m)
        return mse
    
    def gradient(self, X, y, w, b):
        m = X.shape[0]
        err = np.transpose(self.predict(X, w, b) - y) # 1xm
        dj_db = np.sum(err) / m
        dj_dw = np.dot(err, X) / m
        return dj_db, dj_dw
            
    def gradient_descent(self, X, y, w, b, lr=ALPHA, n_iters=MAX_ITERS,
                         cost_fn=None, grad_fn=None):
        if not grad_fn:
            grad_fn = self.gradient
        if not cost_fn:
            cost_fn = self.MSE
        # For displaying stuff - probably delete after
        # TODO
        J_hist = []
        # TODO: Limit iters if diff too low or whatever
        for i in range(n_iters):
            db, dw = grad_fn(X, y, w, b)
            w = w - (lr * dw) # TODO: broadcast fuckery, no -=, Workout the why
            b -= lr * db
            J_hist.append(cost_fn(X, y, w, b))
            if (i % (n_iters / 10) == 0):
                print(f"Iteration {i:4d}:\n\tCost:\t{J_hist[-1]}")
        return w, b, J_hist
    