import numpy as np
import pandas as pd


def sigmoid(x, derivative=False):
    if derivative:
        return x * (1 - x)

    return 1 / (1 + np.exp(-x))


def relu(x, derivative=False):
    y = x.copy()
    if derivative:
        y[x > 0] = 1
        y[x <= 0] = 0.001
        return y

    y[x <= 0] = y[x <= 0] * 0.001
    return y


def mean_square_error(hypothesis, desired, derivative=False):
    if derivative:
        return hypothesis - desired

    loss = hypothesis - desired
    cost = np.sum(loss ** 2) / 2

    return cost


def add_bias(x, bias):
    return np.concatenate((x, bias), axis=1)


def extend_input_bias(x):
    bias = np.ones((len(x), 1), dtype=x.dtype)

    return add_bias(x, bias)


def activate(x, w, activate_fn=None):
    x = extend_input_bias(x)
    y = x.dot(w)

    if activate_fn:
        y = activate_fn(y)

    return y


def gradient_descend(x, w, y, parent_gradient, activate_fn=None):
    x = extend_input_bias(x).T
    delta_y = np.ones(len(y))

    if activate_fn:
        delta_y = activate_fn(y, derivative=True)

    gradient = delta_y * parent_gradient
    delta_w = x.dot(gradient)

    w_without_bias = w[:-1]
    gradient_to_child = gradient.dot(w_without_bias.T)

    return delta_w, gradient_to_child


def place_holder(size):
    return np.array(size, dtype=np.float32)


def generate_weight(number_of_input, number_of_node):
    # number of input + 1 bias
    # weight = np.random.random((number_of_input, number_of_node)) * np.sqrt(2 / (number_of_input + number_of_node))
    # bias = np.zeros((1, number_of_node))
    # return np.concatenate((weight, bias))
    return np.random.random((number_of_input + 1, number_of_node))


def or_operator_test():
    x = np.array([
        [1, 1],
        [1, 0],
        [0, 1],
        [0, 0],
    ])

    y = np.array([
        [0, 1],
        [1, 0],
        [1, 0],
        [0, 1]
    ])

    w1 = generate_weight(2, 5)
    w2 = generate_weight(5, 2)

    learning_rate = 1e0
    epoch = 50000

    ac_fn = sigmoid
    for i in range(epoch):
        h1 = activate(x, w1, activate_fn=ac_fn)
        o = activate(h1, w2, activate_fn=ac_fn)

        error = mean_square_error(o, y)
        print(f"epoch {i} = error:{error}")
        if error < 1e-2:
            break
        delta_error = mean_square_error(o, y, derivative=True)

        delta_w2, gradient = gradient_descend(h1, w2, o, delta_error, activate_fn=ac_fn)
        delta_w1, _ = gradient_descend(x, w1, h1, gradient, activate_fn=ac_fn)

        w1 += -learning_rate * delta_w1
        w2 += -learning_rate * delta_w2

    h1 = activate(x, w1, activate_fn=ac_fn)
    o = activate(h1, w2, activate_fn=ac_fn)

    print(w1)
    print(w2)
    print("Eval", o)


if __name__ == '__main__':
    df = pd.read_csv('data/HTRU_2.csv', header=None)
    print(df.describe())

