from scipy.optimize import leastsq, minimize
import numpy as np


def get_optimization_args(X, Y, U, approximation_args):
    approximation_args["X"] = X
    approximation_args["Y"] = Y
    approximation_args["U"] = U

    return approximation_args


class LinearAlgorithmOptimization:

    def __init__(self, **kwargs):

        self.k = kwargs["k"]
        self.sigma = kwargs["sigma"]
        self.uniform_directions = kwargs["uniform_directions"]
        self.high_lim = kwargs["high_lim"]
        self.x_0_init_random = kwargs["x_0_init_random"]
        self.seed = kwargs["seed"]
        self.U = kwargs["U"]
        self.Y = kwargs["Y"]
        self.X = kwargs["X"]

    def objective(self, X):
        return -X.dot(self.U)

    def constrainet_ij(self, X, *args):
        i, j = args
        x_i = X[2 * i:2 * (i + 1)]
        u_i = self.U[2 * i:2 * (i + 1)]
        x_j = X[2 * j:2 * (j + 1)]

        return x_i.dot(u_i) - x_j.dot(u_i)

    def constrainet_y_ij(self, X, *args):
        i, i = args

        x_i = X[2 * i:2 * (i + 1)]
        u_i = self.U[2 * i:2 * (i + 1)]
        y_i = self.Y[i]

        return y_i - x_i.dot(u_i)

    def get_constrainetes(self):

        consts = []
        for i in range(self.X.shape[0]):

            const_yi_args = {"type": "ineq", "fun": self.constrainet_y_ij, "args": (i, i)}
            consts.append(const_yi_args)

            for j in range(self.X.shape[0]):
                const_xi_args = {"type": "ineq", "fun": self.constrainet_ij, "args": (i, j)}
                consts.append(const_xi_args)

        return consts

    def optimize(self):
        consts = self.get_constrainetes()
        sol = minimize(self.objective, self.X, constraints=consts)

        return sol


class LeastSquares:

    def __init__(self, **kwargs):

        self.k = kwargs["k"]
        self.sigma = kwargs["sigma"]
        self.uniform_directions = kwargs["uniform_directions"]
        self.high_lim = kwargs["high_lim"]
        self.x_0_init_random = kwargs["x_0_init_random"]
        self.seed = kwargs["seed"]
        self.U = kwargs["U"]
        self.Y = kwargs["Y"]
        self.X = kwargs["X"]

    def objective(self, X):
        return (np.sum(Y) - X.dot(self.U)) ** 2

    def constrainet_ij(self, X, *args):
        i, j = args
        x_i = X[2 * i:2 * (i + 1)]
        u_i = self.U[2 * i:2 * (i + 1)]
        x_j = X[2 * j:2 * (j + 1)]

        return x_i.dot(u_i) - x_j.dot(u_i)

    def constrainet_y_ij(self, X, *args):
        i, i = args

        x_i = X[2 * i:2 * (i + 1)]
        u_i = self.U[2 * i:2 * (i + 1)]
        y_i = self.Y[i]

        return y_i - x_i.dot(u_i)

    def get_constrainetes(self):

        consts = []
        for i in range(self.X.shape[0]):

            #             const_yi_args = {"type": "ineq", "fun": self.constrainet_y_ij, "args": (i, i)}
            #             consts.append(const_yi_args)

            for j in range(self.X.shape[0]):
                const_xi_args = {"type": "ineq", "fun": self.constrainet_ij, "args": (i, j)}
                consts.append(const_xi_args)

        return consts

    def optimize(self):
        consts = self.get_constrainetes()
        sol = minimize(self.objective, self.X, constraints=consts)

        return sol
