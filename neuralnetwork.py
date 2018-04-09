import numpy as np


def mean_square_error(hypothesis, desired, derivative=False):
    if derivative:
        return hypothesis - desired

    loss = hypothesis - desired
    cost = np.sum(loss ** 2) / (2 * desired.shape[0])

    return cost


def add_bias(x, bias):
    return np.concatenate((x, bias), axis=1)


def extend_input_bias(x):
    bias = np.ones((len(x), 1), dtype=x.dtype)

    return add_bias(x, bias)


class Layer(object):
    def forward(self, input):
        """
        :param input:
        :return: output of this layer
        """
        raise NotImplementedError

    def backward(self, input, gradient, learning_rate):
        """
        :param input, gradient, learning_rate:
        :return: gradient
        """
        raise NotImplementedError


class LinearLayer(Layer):
    def __init__(self, number_of_input, number_of_node):
        self.weight = np.random.randn(number_of_input + 1, number_of_node)

    def forward(self, input):
        input = extend_input_bias(input)
        return input.dot(self.weight)

    def backward(self, input, gradient, learning_rate):
        input = extend_input_bias(input).T
        delta_w = input.dot(gradient)

        self.weight += -learning_rate * delta_w

        return gradient.dot(self.weight[:-1].T)


class SigmoidActivation(Layer):
    @staticmethod
    def _sigmoid(x, derivative=False):
        if derivative:
            return x * (1 - x)

        return 1 / (1 + np.exp(-x))

    def forward(self, input):
        return self._sigmoid(input)

    def backward(self, input, gradient, learning_rate=None):
        output = self.forward(input)
        return self._sigmoid(output, derivative=True) * gradient


class Input(Layer):
    def __init__(self):
        self.variable = None

    def set(self, input):
        self.variable = input

    def forward(self, input):
        return self.variable

    def backward(self, input, gradient, learning_rate):
        pass


def forward(layers):
    input = layers[0]
    outputs = [input.forward(None)]

    for layer in layers[1:]:
        x = outputs[-1]
        y = layer.forward(x)
        outputs.append(y)

    return outputs


def backward(outputs, layers, error_gradient, learning_rate=1e-1):
    gradient = error_gradient
    outputs.pop()
    for layer in reversed(layers[1:]):
        output = outputs.pop()
        gradient = layer.backward(output, gradient, learning_rate)


if __name__ == '__main__':
    input = Input()

    layers = [
        input
        , LinearLayer(2, 5)
        , SigmoidActivation()
        , LinearLayer(5, 2)
        , SigmoidActivation()
    ]

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

    input.set(x)
    epoch = 300
    learning_rate = 3

    i = 1
    error = 10
    while error > 1e-2:
        outputs = forward(layers)
        y_hat = outputs[-1]

        error = mean_square_error(y_hat, y)
        print(f"{i} Error {error}")

        error_gradient = mean_square_error(y_hat, y, derivative=True)

        backward(outputs, layers, error_gradient, learning_rate)

        i += 1

    print(y)
    print(y_hat)
