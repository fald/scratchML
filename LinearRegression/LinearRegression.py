import numpy as np
from config import *

# TODO: Fill out class
# TODO: Tests
# TODO: docstrings
# TODO: error/sanity checking
# TODO: Regularization
# TODO: Normal Equation solution
# TODO: Over/under fit and common solutions, automatic
# TODO: lol w, b should just be instance vars

class LinearRegression:
    def __init__(self):
        self.w = None
        self.b = None
    
    def fit(self, X, y, init_random=False):
        m, n = X.shape
        self.init_weights(n, init_random)
        self.gradient_descent(X, y, history=True)
    
    def predict(self, X):
        return np.dot(X, self.w) + self.b
    
    def init_weights(self, n, random=False):
        if not random:
            self.w = np.zeros(n)
            self.b = 0
        else:
            # TODO: This branch
            pass
    
    def MSE(self, X, y):        
        m = X.shape[0]
        err = self.predict(X) - y
        sq_err = np.power(err, 2)
        mse = np.sum(sq_err) / (2 * m)
        return mse
    
    def gradient(self, X, y):
        m = X.shape[0]
        err = np.transpose(self.predict(X) - y) # 1xm
        dj_db = np.sum(err) / m
        dj_dw = np.dot(err, X) / m
        return dj_db, dj_dw
            
    def gradient_descent(self, X, y, lr=ALPHA, n_iters=MAX_ITERS,
                         cost_fn=None, grad_fn=None, history=False):
        if not grad_fn:
            grad_fn = self.gradient
        if not cost_fn:
            cost_fn = self.MSE
        # For displaying stuff - probably delete after
        # TODO
        J_hist = []
        # TODO: Limit iters if diff too low or whatever
        for i in range(n_iters):
            db, dw = grad_fn(X, y)
            self.w = self.w - (lr * dw) # TODO: broadcast fuckery, no -=, Workout the why
            self.b -= lr * db
            if history:
                J_hist.append(cost_fn(X, y))
                if (i % (n_iters / 10) == 0) or (i == n_iters - 1):
                    print(f"Iteration {i:4d}:\n\tCost:\t{J_hist[-1]}")
    