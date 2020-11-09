import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from linearRegression import LinearRegersion


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4))
    fig.set_tight_layout(True)

    # Set up the dataset
    np.random.seed(80)
    X = np.arange(1, 10, 0.1)
    y = np.random.normal(np.log(np.power(X, 5)), 1.3, len(X))
    ax1.scatter(X, y)

    iters = 200  # Iterations that Gradient Descent loops through
    learning_rate = 0.0015  # Learning rate for Gradient Descent

    lr = LinearRegersion(X, y, learning_rate)

    line, = ax1.plot(X, X, 'r-', linewidth=2)

    def update(i):
        lr.gradient()
        label = f'Iteration: {i + 1}'
        thetas = lr.get_theta()
        ax1.set_xlabel(label)
        line.set_ydata(X * thetas[1] + thetas[0])

        ax2.scatter(i, lr.cost())
        ax2.set_ylabel("Value of Cost")
        ax2.set_xlabel("Iteration")

        return line, ax1

    anim = FuncAnimation(fig, update, repeat=False, frames=iters, interval=30)

    plt.show()


if __name__ == '__main__':
    main()
