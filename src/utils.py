from scipy.spatial.distance import directed_hausdorff
import numpy as np
from numpy import *
import matplotlib.pyplot as plt

from data_generator import DataGenerator
from optimize import LinearAlgorithmOptimization, LeastSquares, get_optimization_args


def plot_housdorf_distance_vs_k(body, **args):
    us_range_ = args["k_range"]
    step_size = args["step_size"]
    sigma = args["sigma"]
    uniform_directions = args["uniform_directions"],
    high_lim = args["high_lim"],
    x_0_init_random = args["x_0_init_random"],
    seed = args["seed"]

    datagen = DataGenerator(d=2)

    hous_dists = []
    for i in range(1, us_range_, 5):
        approximation_args = {
            "k": i,
            "sigma": sigma,
            "uniform_directions": uniform_directions,
            "high_lim": high_lim,
            "x_0_init_random": x_0_init_random,
            "seed": seed
        }

        us = datagen.generate_us(**approximation_args)
        noise_supports = datagen.generate_noise_support(body=body, us=us, sigma=sigma)
        Y, U, X = datagen.get_optimization_params(noise_supports, us, x_random_init=x_0_init_random)
        optimization_args = get_optimization_args(X, Y, U, approximation_args)
        solver = LinearAlgorithmOptimization(**optimization_args)
        solution = solver.optimize()
        X_ = solution.x
        X_ = X_.reshape(-1, 2)
        dist, ind_1, ind_2 = directed_hausdorff(body.vertices, X_)
        hous_dists.append(dist)

    fig = plt.figure(figsize=(12, 8))
    plt.plot(list(range(1, us_range_, step_size)), hous_dists)
    plt.xlabel("Number off measuerments")
    plt.ylabel("Hausdorf distance")
    plt.show()
    return hous_dists


def plot_housdorf_distance_vs_noise(body, **args):
    sigma_range = args["sigma_range"]
    step_size = args["step_size"]
    k = args["k"]
    uniform_directions = args["uniform_directions"],
    high_lim = args["high_lim"],
    x_0_init_random = args["x_0_init_random"],
    seed = args["seed"]

    datagen = DataGenerator(d=2)

    hous_dists = []
    for i in np.arange(0, sigma_range, step_size):
        sigma = i
        approximation_args = {
            "k": k,
            "sigma": sigma,
            "uniform_directions": uniform_directions,
            "high_lim": high_lim,
            "x_0_init_random": x_0_init_random,
            "seed": seed
        }

        us = datagen.generate_us(**approximation_args)
        noise_supports = datagen.generate_noise_support(body=body, us=us, sigma=sigma)
        Y, U, X = datagen.get_optimization_params(noise_supports, us, x_random_init=x_0_init_random)
        optimization_args = get_optimization_args(X, Y, U, approximation_args)
        solver = LinearAlgorithmOptimization(**optimization_args)
        solution = solver.optimize()
        X_ = solution.x
        X_ = X_.reshape(-1, 2)
        dist, ind_1, ind_2 = directed_hausdorff(body.vertices, X_)
        hous_dists.append(dist)

    fig = plt.figure(figsize=(12, 8))
    plt.plot(list(np.arange(0, sigma_range, step_size)), hous_dists)
    plt.xlabel("Noise level")
    plt.ylabel("Hausdorf distance")
    plt.show()
    return hous_dists


def plot_lin_vs_lsq(body, **args):
    us_range_ = args["k_range"]
    step_size = args["step_size"]
    sigma = args["sigma"]
    uniform_directions = args["uniform_directions"],
    high_lim = args["high_lim"],
    x_0_init_random = args["x_0_init_random"],
    seed = args["seed"]

    datagen = DataGenerator(d=2)

    hous_dists_lin = []
    hous_dists_lst_sq = []
    for i in range(1, us_range_, 5):
        approximation_args = {
            "k": i,
            "sigma": sigma,
            "uniform_directions": uniform_directions,
            "high_lim": high_lim,
            "x_0_init_random": x_0_init_random,
            "seed": seed
        }

        us = datagen.generate_us(**approximation_args)
        noise_supports = datagen.generate_noise_support(body=polygon, us=us, sigma=sigma)
        Y, U, X = datagen.get_optimization_params(noise_supports, us, x_random_init=x_0_init_random)
        optimization_args = get_optimization_args(X, Y, U, approximation_args)
        solver = LinearAlgorithmOptimization(**optimization_args)
        solution_linear = solver.optimize()
        X_linear = solution_linear.x
        X_linear = X_linear.reshape(-1, 2)
        dist_linear, ind_1, ind_2 = directed_hausdorff(body.vertices, X_linear)
        hous_dists_lin.append(dist_linear)

        lst_squares = LeastSquares(**optimization_args)
        solution_lst = lst_squares.optimize()
        X_lst = solution_lst.x
        X_lst = X_lst.reshape(-1, 2)
        dist_lst, ind_1, ind_2 = directed_hausdorff(body.vertices, X_lst)
        hous_dists_lst_sq.append(dist_lst)

    fig = plt.figure(figsize=(12, 8))
    plt.plot(list(range(1, us_range_, step_size)), hous_dists_lin, label="Linear algorithm")
    plt.plot(list(range(1, us_range_, step_size)), hous_dists_lst_sq, label="Least squares")
    plt.legend()
    plt.xlabel("Number off measuerments")
    plt.ylabel("Hausdorf distance")
    plt.show()
    return 1
