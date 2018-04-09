import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import dataset
import neuralnetwork as nn


def preprocess(input_matrix):
    pass


def accuracy_score(hypothesis, desired):
    hypothesis[hypothesis > 0.6] = 1
    hypothesis[hypothesis < 0.4] = 0

    return np.sum(hypothesis == desired) / len(desired)


def create_nn():
    return [
        nn.Input()
        , nn.LinearLayer(8, 16)
        , nn.SigmoidActivation()
        , nn.LinearLayer(16, 16)
        , nn.SigmoidActivation()
        , nn.LinearLayer(16, 8)
        , nn.SigmoidActivation()
        , nn.LinearLayer(8, 1)
        , nn.SigmoidActivation()
    ]


if __name__ == '__main__':
    dataset.prepare_dataset('data')
    df = pd.read_csv('data/HTRU_2.csv', header=None, names=['1', '2', '3', '4', '5', '6', '7', '8', 'class'])

    input_matrix = df[['1', '2', '3', '4', '5', '6', '7', '8']].as_matrix()
    class_label = df[['class']].as_matrix()

    print(f"class 0: {np.sum(class_label == 0)}, 1: {np.sum(class_label == 1)}")

    kf = KFold(n_splits=10)
    accuracy_list = []
    highest_accuracy = 0
    model = None
    for train_index, test_index in kf.split(input_matrix):
        input_train, class_train = input_matrix[train_index], class_label[train_index]
        input_test, class_test = input_matrix[test_index], class_label[test_index]

        nn_layers = create_nn()
        nn_layers[0].set(input_train)

        error = np.inf
        acceptable_error = 1e-2
        epoch = 1
        learning_rate = 1e-4
        max_epoch = 1000
        while error > acceptable_error and epoch < max_epoch + 1:
            outputs = nn.forward(nn_layers)
            hypothesis = outputs[-1]

            error = nn.mean_square_error(hypothesis, class_train)
            error_gradient = nn.mean_square_error(hypothesis, class_train, derivative=True)

            nn.backward(outputs, nn_layers, error_gradient, learning_rate)

            epoch += 1

        nn_layers[0].set(input_test)
        outputs = nn.forward(nn_layers)
        hypothesis = outputs[-1]

        accuracy = accuracy_score(hypothesis, class_test)

        if highest_accuracy < accuracy:
            highest_accuracy = accuracy
            model = nn_layers

        print(f"Accuracy {accuracy}")
        accuracy_list.append(accuracy)

    print("Average Accuracy: ", sum(accuracy_list) / len(accuracy_list))

    for layer in nn_layers:
        if type(layer) is nn.LinearLayer:
            print("W", layer.weight)
