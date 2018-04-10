import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import KFold
import seaborn as sns

import dataset
import neuralnetwork as nn


def log_scale(x):
    return np.log1p(x)


def minmax_scale(x, min, max):
    pass


def preprocess(input_matrix):
    input_matrix[:, 7] = log_scale(input_matrix[:, 7])


def translate_class(x):
    x = x.copy()

    return np.argmax(x, axis=1)


def f1_score(y_predict, y_true):
    return metrics.f1_score(y_true, y_predict)


def accuracy_score(y_predict, y_true):
    return np.sum(y_predict == y_true) / len(y_true)


def confusion_matrix(y_predict, y_true):
    # http://peterroelants.github.io/posts/neural_network_implementation_part05/
    conf_matrix = metrics.confusion_matrix(y_true, y_predict, labels=None)  # Get confustion matrix
    # Plot the confusion table
    class_names = ['${:d}$'.format(x) for x in range(0, 10)]  # Digit class names
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # Show class labels on each axis
    ax.xaxis.tick_top()
    major_ticks = range(0, 10)
    minor_ticks = [x + 0.5 for x in range(0, 10)]
    ax.xaxis.set_ticks(major_ticks, minor=False)
    ax.yaxis.set_ticks(major_ticks, minor=False)
    ax.xaxis.set_ticks(minor_ticks, minor=True)
    ax.yaxis.set_ticks(minor_ticks, minor=True)
    ax.xaxis.set_ticklabels(class_names, minor=False, fontsize=15)
    ax.yaxis.set_ticklabels(class_names, minor=False, fontsize=15)
    # Set plot labels
    ax.yaxis.set_label_position("right")
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    fig.suptitle('Confusion table', y=1, fontsize=15)
    # Show a grid to seperate digits
    ax.grid(b=True, which=u'minor')
    # Color each grid cell according to the number classes predicted
    ax.imshow(conf_matrix, interpolation='nearest', cmap='binary')
    # Show the number of samples in each cell
    for x in range(conf_matrix.shape[0]):
        for y in range(conf_matrix.shape[1]):
            color = 'g' if x == y else 'k'
            ax.text(x, y, conf_matrix[y, x], ha="center", va="center", color=color)
    plt.show()


def create_nn():
    return [
        nn.Input()
        , nn.LinearLayer(8, 16)
        , nn.LogisticActivation()
        , nn.LinearLayer(16, 16)
        , nn.LogisticActivation()
        , nn.LinearLayer(16, 16)
        , nn.LogisticActivation()
        , nn.LinearLayer(16, 2)
        , nn.SoftmaxActivation()
    ]


def train_batch(x, y, nn_layers, learning_rate):
    nn_layers[0].set(x)
    outputs = nn.forward(nn_layers)
    y_predict = outputs[-1]

    nn.backward(outputs, nn_layers, y, learning_rate)

    return nn.cross_entropy(y_predict, y)


def train_stochastic(x, y, nn_layers, learning_rate):
    size = x.shape[0]
    idx = np.arange(size)
    np.random.shuffle(idx)
    loss = 0

    for i in idx:
        input_train = np.array([x[i]])
        class_train = np.array([y[i]])
        loss += train_batch(input_train, class_train, nn_layers, learning_rate)

    return loss / size


def train_minibatch(x, y, nn_layers, learning_rate, batch_size=10):
    n_batch = x.shape[0] / batch_size
    batches = zip(
        np.array_split(x, n_batch, axis=0)
        , np.array_split(y, n_batch, axis=0)
    )
    loss = 0
    for x, y in batches:
        loss += train_batch(x, y, nn_layers, learning_rate)

    return loss / n_batch


def predict(x, nn_layers):
    nn_layers[0].set(x)
    outputs = nn.forward(nn_layers)
    y_predict = outputs[-1]

    return y_predict


if __name__ == '__main__':
    dataset.prepare_dataset('data')
    df = pd.read_csv('data/HTRU_2.csv', header=None, names=['1', '2', '3', '4', '5', '6', '7', '8', 'class'])

    print(df.describe())
    # features = ['1', '2', '3', '4', '5', '6', '7', '8']
    # for feature in features:
    #     df[[feature]].boxplot()
    #
    #     plt.show()
    input_matrix = df[['1', '2', '3', '4', '5', '6', '7', '8']].as_matrix()

    preprocess(input_matrix)
    raw_class_label = df[['class']].as_matrix()
    #
    # class_label = np.zeros((raw_class_label.shape[0], 2))
    # class_label[np.arange(raw_class_label.shape[0]), raw_class_label[:, 0]] = 1
    #
    # print(class_label)
    #
    # print(f"class 0: {np.sum(np.argmax(class_label, axis=1) == 0)}, 1: {np.sum(np.argmax(class_label, axis=1) == 1)}")
    #
    # kf = KFold(n_splits=8)
    # learning_rate = 1e-1
    # batch_size = 100
    # max_epoch = 300
    # acceptable_loss = 1e-2
    #
    # score_list = []
    # highest_score = 0
    # model = None
    # for train_index, test_index in kf.split(input_matrix):
    #     input_train, class_train = input_matrix[train_index], class_label[train_index]
    #     input_test, class_test = input_matrix[test_index], class_label[test_index]
    #
    #     nn_layers = create_nn()
    #
    #     cost = np.inf
    #     moving_cost = np.inf
    #     epoch = 0
    #     cost_list = []
    #     while epoch <= max_epoch and moving_cost > 1e-4:
    #         new_cost = train_minibatch(input_train, class_train, nn_layers, learning_rate, batch_size=batch_size)
    #
    #         if np.isinf(cost):
    #             moving_cost = cost = new_cost
    #         else:
    #             old_cost = cost
    #             cost = 0.75 * cost + 0.25 * new_cost
    #             moving_cost = 0.75 * moving_cost + 0.25 * np.abs(old_cost - cost)
    #
    #         cost_list.append(cost)
    #         if epoch % 10 == 0:
    #             output = predict(input_test, nn_layers)
    #
    #             score = f1_score(translate_class(output), translate_class(class_test))
    #             print(f"Epoch {epoch} | cost: {cost}, moving cost: {moving_cost}, f1 score: {score}")
    #
    #         epoch += 1
    #
    #     print(f"Epoch {epoch} | cost: {cost}, moving cost: {moving_cost}, f1 score: {score}")
    #
    #     output = predict(input_test, nn_layers)
    #     score = f1_score(translate_class(output), translate_class(class_test))
    #     confusion_matrix(translate_class(output), translate_class(class_test))
    #
    #     plt.plot(cost_list)
    #     plt.show()
    #
    #     if highest_score < score:
    #         highest_score = score
    #         model = nn_layers
    #
    #     print(f"Score: {score}")
    #
    #     score_list.append(score)
    #
    # print("Average Score: ", sum(score_list) / len(score_list))
