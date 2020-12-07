import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


# from https://stackoverflow.com/questions/22867620/putting-arrowheads-on-vectors-in-matplotlibs-3d-plot
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


class DataGenerator():

    def __init__(self, d):
        if d != 2 and d != 3:
            raise ValueError("Argument d must be 2 or 3")
        else:
            self.d = d

    def generate_us(self, **kwargs):

        self.k = kwargs["k"]
        self.uniform_directions = kwargs["uniform_directions"]
        self.high_lim = kwargs["high_lim"]
        self.seed = kwargs["seed"]

        if self.d == 2:
            # checking dimension
            if self.uniform_directions:
                thetas = np.linspace(0, self.high_lim, self.k)
            else:
                np.random.seed = self.seed
                thetas = np.random.uniform(0, self.high_lim, self.k)

            angles = np.array([np.array([np.cos(theta), np.sin(theta)]).reshape(-1, 1) for theta in thetas])
        elif self.d == 3:

            if self.uniform_directions:
                thetas = np.linspace(0, self.high_lim, self.k)
                phis = np.linspace(0, np.pi / 2.0, self.k)
            else:
                np.random.seed = self.seed
                thetas = np.random.uniform(0, self.high_lim, self.k)
                phis = np.linspace(0, np.pi / 4.0, self.k)

            angles = np.array(
                [np.array([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)]).reshape(-1, 1) for
                 theta, phi in
                 zip(thetas, phis)])

        return np.array(angles)

    def plot_us(self, us):
        us_distribution = "Uniformly" if self.uniform_directions else "Randomly"

        if self.d == 2:
            plt.figure(figsize=(12, 8))
            for i in range(us.shape[0]):
                plt.scatter(us[i][0], us[i][1])

            plt.title("{0} generated {1} directions".format(us_distribution, self.k))
            plt.show()
        else:
            # from https://stackoverflow.com/questions/22867620/putting-arrowheads-on-vectors-in-matplotlibs-3d-plot

            fig = plt.figure(figsize=(15, 15))
            ax = fig.add_subplot(111, projection='3d')

            for u in us:
                ax.plot(u[0], u[1], u[2], 'o', markersize=10, color='g', alpha=0.2)
                # ax.plot([mean_x], [mean_y], [mean_z], 'o', markersize=10, color='red', alpha=0.5)
                a = Arrow3D([u[0].item(), 0], [u[1].item(), 0],
                            [u[2].item(), 0], mutation_scale=20,
                            lw=3, arrowstyle="-|>", color="r")
                ax.add_artist(a)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')

            plt.title("{0} generated {1} directions".format(us_distribution, self.k))

            plt.draw()
            plt.show()

    def generate_supports(self, body, us):
        return [body.support(u=u) for u in us]

    def generate_noise_support(self, body, us, sigma):
        self.sigma = sigma
        return [body.support(u=u) + np.random.normal(0, sigma) for u in us]

    def plot_noise_supports(self, us, noise_supports):
        if self.d == 2:
            plt.figure(figsize=(12, 8))
            for i in range(us.shape[0]):
                point = us[i] * noise_supports[i]
                plt.scatter(point[0], point[1])
            plt.title("Noise support vectors for k = {0},sigma = {1}".format(self.k, self.sigma))

            plt.show()

        elif self.d == 3:
            noise_support_vectors = np.expand_dims(us[:, :, 0] * np.array(noise_supports).reshape(-1, 1),
                                                   axis=2).astype("float64")
            self.plot_us(noise_support_vectors)

    @staticmethod
    def get_optimization_params(noise_supports, us, x_random_init, d=2):
        Y = np.array(noise_supports).reshape(-1, 1)
        U = us[:, :, 0]
        U = U.flatten()
        if d == 2:
            if x_random_init:
                X = np.random.randn(us.shape[0], 2)
            else:
                X = np.zeros((us.shape[0], 2))
        else:
            if x_random_init:
                X = np.random.randn(us.shape[0], 3)
            else:
                X = np.zeros((us.shape[0], 3))
        return Y, U, X


if __name__ == "__main__":
    D = 3
    CENTROID = [0, 0]
    NUM_VERTICES = 4
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

    datagen = DataGenerator(d=D)
    us = datagen.generate_us(**approximation_args)

    noise_supports = datagen.generate_noise_support(body=polyhedron, us=us, sigma=SIGMA)
    datagen.plot_noise_supports(us, noise_supports)
