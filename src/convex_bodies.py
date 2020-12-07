import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from numpy import pi
from collections import defaultdict
from abc import ABC, abstractmethod
from pprint import pprint
from sympy import *


import sys

sys.path.insert(1, '../src/pyConvexHull3D/')
from hull3D import ConvexHull3D


class ConvexBody(ABC):
    """
    Abstract class for convex bodies

    """

    def __init__(self, body_type):
        super().__init__()

    def equation(self, point):
        raise NotImplementedError

    def is_edge(self, point):
        raise NotImplementedError

    def edge(self, n):
        raise NotImplementedError

    def is_interior(self, point):
        raise NotImplementedError

    #     @abstractmethod
    def interior(self, n):
        pass

    @abstractmethod
    def support(self, u):
        pass

    @abstractmethod
    def support_vector(self, u):
        pass

    #     @abstractmethod
    def plot_body(self, show=False):
        pass

    def convex_hull(self):
        return ConvexHull(self.interior_points)


class Circle(ConvexBody):
    """
    Cirlcle class inherits from ConvexBody

    """

    def __init__(self, center, R):
        self.center = center
        self.R = R
        self.body_type = "circle"

    def equation(self, point):
        x, y = point[0], point[1]
        return x ** 2 + y ** 2

    def edge(self, n):
        return self.center + np.array(
            [(np.cos(2 * pi / n * x) * self.R, np.sin(2 * pi / n * x) * self.R) for x in range(0, n + 1)])

    def is_edge(self, point):
        if self.equation(point) == self.R ** 2:
            return True
        else:
            return False

    def is_interior(self, point):
        if self.equation(point) < self.R ** 2:
            return True
        else:
            return False

    def generate_interior_points(self, n):
        self.interior_points = []
        for i in range(n):
            x, y = np.random.uniform(-self.R, self.R), np.random.uniform(-self.R, self.R)
            if self.is_interior(point=[x, y]):
                self.interior_points.append([x, y])
            else:
                continue
        self.interior_points = np.array(self.interior_points)
        return self.interior_points

    def support(self, u):
        return 2 * self.R


class Polyhedron(ConvexBody):

    def __init__(self, args):
        self.body_type = "polyhedron"
        super().__init__(self.body_type)

        self.n = args["num_vertices"]
        self.vertices = args["vertices"]
        self.right = args["right"]
        self.centroid = args["centroid"]

    @staticmethod
    def get_projection(x, u):
        norms = np.dot(x, u)
        vectors = np.array([norms[i][0] * u for i in range(norms.shape[0])]).reshape(norms.shape[0], u.shape[0])
        return norms, vectors

    #
    @staticmethod
    def sort_projections(norms, vectors):
        return norms[np.argsort(norms[:, 0])], vectors[np.argsort(norms[:, 0])]

    #
    def support_vertice(self, u):
        norms, vectors = self.get_projection(x=self.vertices, u=u)
        return self.vertices[np.asscalar(np.argmax(norms, axis=0))]

    #
    def support(self, u):
        norms, vectors = self.get_projection(x=self.vertices, u=u)
        return np.asscalar(np.max(norms, axis=0))

    #
    def support_vector(self, u):
        norms, vectors = self.get_projection(x=self.vertices, u=u)
        return vectors[np.argmax(norms, axis=0)].reshape(2, 1)

    #
    def plot_body(self, show=True):
        # pts = (np.ones((num_vertices, 3)) * centroid + np.random.randn(num_vertices, 3)).astype('float64')
        pts = np.array(self.vertices).reshape(-1,3).astype("float64")

        c_hull = ConvexHull3D(pts, run=True, preproc=False, make_frames=False, frames_dir='./frames/')

        # self.vertices = c_hull.DCEL.vertexDict.values()
        c_hull.generateImage(show=show)



class Polygon(ConvexBody):

    def __init__(self, args):
        self.body_type = "polygon"
        super().__init__(self.body_type)

        self.n = args["num_vertices"]
        self.vertices = args["vertices"]
        self.right = args["right"]
        self.centroid = args["centroid"]

    @staticmethod
    def get_projection(x, u):
        norms = np.dot(x, u)
        vectors = np.array([norms[i][0] * u for i in range(norms.shape[0])]).reshape(norms.shape[0], u.shape[0])
        return norms, vectors

    @staticmethod
    def sort_projections(norms, vectors):
        return norms[np.argsort(norms[:, 0])], vectors[np.argsort(norms[:, 0])]

    def support_vertice(self, u):
        norms, vectors = self.get_projection(x=self.vertices, u=u)
        return self.vertices[np.asscalar(np.argmax(norms, axis=0))]

    def support(self, u):
        norms, vectors = self.get_projection(x=self.vertices, u=u)
        return np.asscalar(np.max(norms, axis=0))

    def support_vector(self, u):
        norms, vectors = self.get_projection(x=self.vertices, u=u)
        return vectors[np.argmax(norms, axis=0)].reshape(2, 1)

    def plot_body(self, show=False):

        vertices = np.array(self.vertices)
        c_hull = ConvexHull(vertices)
        plt.figure(figsize=(12, 8))
        for simplex in c_hull.simplices:
            plt.plot(vertices[simplex, 0], vertices[simplex, 1], 'b-')
        plt.title("Generated Polygon")
        if show:
            plt.show()

    def plot_support(self, u):
        h_u = self.support(u=u)
        h_vec = self.support_vector(u=u)
        # support_vertice = self.support_vertice(u=u)
        h_u_norm = h_vec / h_u
        tan_phi = np.asscalar(u[1] / u[0])

        h_u_vec_for_plot = list(map(lambda x: float(x), [0, 0, np.asscalar(h_u_norm[0]), np.asscalar(h_u_norm
                                                                                                     [1])]))
        # support_vertice_vector = list(map(lambda x: float(x), [np.asscalar(h_u_vec[0]),
        #                                                        np.asscalar(h_u_vec[1]),
        #                                                        support_vertice[0],
        #                                                        support_vertice[1]]))

        x_right_lim = self.support(u=np.array([np.cos(0), np.sin(0)]).reshape(-1, 1)) + 1
        x_left_lim = -self.support(u=np.array([np.cos(0), np.sin(0)]).reshape(-1, 1)) - 1
        y_right_lim = self.support(u=np.array([np.cos(np.pi / 2.), np.sin(np.pi / 2.0)]).reshape(-1, 1)) + 1
        y_left_lim = -self.support(u=np.array([np.cos(np.pi / 2.), np.sin(np.pi / 2.0)]).reshape(-1, 1)) - 1

        self.plot_body(show=False)

        plt.grid()
        ax = plt.axes()

        ax.arrow(*h_u_vec_for_plot, head_width=0.05, head_length=0.06)
        # ax.arrow(*support_vertice_vector, head_width=0.05, head_length=0.06)

        y_plot = tan_phi * x_right_lim
        plt.plot([0, x_right_lim], [0, y_plot], "k--")

        plt.xlim(int(round(x_left_lim)), int(round(x_right_lim)))
        plt.ylim(int(round(y_left_lim)), int(round(y_right_lim)))

        plt.title('Support Vector for u = {0}'.format(u), fontsize=10)

        plt.show()

    def plot_approximation(self, X_, **args):
        k = args["k"]
        sigma = args["sigma"]

        vertices = np.array(self.vertices)
        c_hull_body = ConvexHull(vertices)

        c_hull_approximation = ConvexHull(X_)

        plt.figure(figsize=(12, 8))
        for simplex in c_hull_body.simplices[:-1, :]:
            plt.plot(vertices[simplex, 0], vertices[simplex, 1], 'b-')

        plt.plot(vertices[c_hull_body.simplices[-1], 0], vertices[c_hull_body.simplices[-1], 1], 'b-',
                 label="Input body")

        for simplex in c_hull_approximation.simplices[:-1, :]:
            plt.plot(X_[simplex, 0], X_[simplex, 1], 'r--')

        plt.plot(X_[c_hull_approximation.simplices[-1], 0], X_[c_hull_approximation.simplices[-1], 1], 'r--',
                 label="Linear algorithm output")

        plt.title("Approximated polygon with k={0} and sigma={1}".format(k, sigma))
        plt.legend()
        plt.show()


class ConvexBodyGenerator(ABC):
    """
    Abstract Class for Convex Body generation
    """

    def __init__(self, body_type):
        super().__init__()
        self.default_args = self.get_default_args()
        self.body_type = body_type

    def get_default_args(self):
        default_dict_keys = ["centroid", "num_vertices", "R", "right"]
        d = dict(zip(default_dict_keys, [None for i in range(len(default_dict_keys))]))
        return d

    def pars_args(self, args):
        body_args = self.default_args.copy()
        for key in args:
            body_args[key] = args[key]
        return body_args

    @abstractmethod
    def generate(self, args):
        raise NotImplementedError


class PolyhedronGenerator(ConvexBodyGenerator):
    """
    GeneratePolygon class inherits from ConvexBodyGenerator

    """

    def __init__(self):
        self.body_type = "polyhedron"
        super().__init__(self.body_type)

    def generate(self, args):
        polygon_args = self.pars_args(args)
        centroid = Point(polygon_args["centroid"])
        num_vertices = polygon_args["num_vertices"]
        random_seed = polygon_args["random_seed"]

        np.random.seed = random_seed

        vertices = [
            np.array([centroid[0] + np.random.randint(0, 15),
                      centroid[1] + np.random.randint(0, 15),
                      centroid[2] + np.random.randint(0, 15)]) for i in range(num_vertices)]

        # print(vertice)
        args["vertices"] = vertices
        poly = Polyhedron(args=args)

        return poly, np.array(vertices)


class GeneratePolygon(ConvexBodyGenerator):
    """
    GeneratePolygon class inherits from ConvexBodyGenerator

    """

    def __init__(self):
        self.body_type = "polygon"
        super().__init__(self.body_type)

    def generate(self, args):
        polygon_args = self.pars_args(args)
        centroid = Point(polygon_args["centroid"])
        num_vertices = polygon_args["num_vertices"]
        random_seed = polygon_args["random_seed"]

        np.random.seed = random_seed

        if args["right"]:
            phis = np.arange(start=0, stop=2 * np.pi, step=2 * np.pi / num_vertices)
            vertices = [[centroid[0] + np.cos(phi), centroid[1] + np.sin(phi)] for phi in phis]

        else:
            vertices = [[centroid[0] + np.random.random(), centroid[1] + np.random.random()] for i in
                        range(num_vertices)]

        args["vertices"] = vertices
        poly = Polygon(args=args)

        return poly, np.array(vertices)


if __name__ == "__main__":
    D = 3
    CENTROID = [0.0, 0.0, 0.0]
    NUM_VERTICES = 25
    RIGHT = True

    K = 80
    UNIFORM_DIRECTIONS = True
    HIGH_LIM = 2 * np.pi
    RANDOM_SEED = 42.0
    SIGMA = 0.1
    X_0_INIT_RANDOM = True

    polygon_args = {
        "centroid": CENTROID,
        "num_vertices": NUM_VERTICES,
        "right": RIGHT,
        "random_seed": RANDOM_SEED
    }

    approximation_args = {
        "k": K,
        "sigma": SIGMA,
        "uniform_directions": UNIFORM_DIRECTIONS,
        "high_lim": HIGH_LIM,
        "x_0_init_random": X_0_INIT_RANDOM,
        "seed": RANDOM_SEED
    }

    polyhedron_generator = PolyhedronGenerator()

    polyhedron, vertices = polyhedron_generator.generate(polygon_args)
    # polyhedron.plot_body()

    # datagen = DataGenerator(d=D)
    us = datagen.generate_us(**approximation_args)

    noise_supports = datagen.generate_noise_support(body=polyhedron, us=us, sigma=SIGMA)
    datagen.plot_noise_supports(us, noise_supports)
    # datagen.plot_us(us)
