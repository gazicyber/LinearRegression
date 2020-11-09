import numpy as np


class LinearRegersion:
    def __init__(self, x_vals, y_vals, learning_rate):
        self.m = x_vals.size
        self.lr = learning_rate
        self.x_vals = np.concatenate((np.ones((self.m, 1)), x_vals.reshape((-1, 1))), axis=1)
        self.y_vals = y_vals.reshape((-1, 1))
        self.theta = np.zeros((2, 1))

    # h(x) = X * theta ---- product operation is matrix multiplication
    def hypothesis(self):
        return self.x_vals.dot(self.theta)

    # error = h(x) - y ---- subtraction operation is element-wise operation
    def error(self):
        return self.hypothesis() - self.y_vals

    # cost = (1 / (2*m)) * sum(error ^ 2) ---- square operation is element-wise operation
    def cost(self):
        return (1 / (2 * self.m)) * np.sum(np.square(self.error()))

    # Partial derivative of cost function = X.transpose() * error  ---- product operation is matrix multiplication
    def gradient(self):
        self.theta = self.theta - self.lr * (1 / self.m) * (np.transpose(self.x_vals).dot(self.error()))

    def get_theta(self):
        return self.theta
