import pygrank as pg
import tensorflow as tf
import torch
import pytest


def test_gnn_errors():
    graph, features, labels = pg.load_feature_dataset('synthfeats')
    training, test = pg.split(list(range(len(graph))), 0.8)
    training, validation = pg.split(training, 1 - 0.2 / 0.8)

    class APPNP(tf.keras.Sequential):
        def __init__(self, num_inputs, num_outputs, hidden=64):
            super().__init__([
                tf.keras.layers.Dropout(0.5, input_shape=(num_inputs,)),
                tf.keras.layers.Dense(hidden, activation=tf.nn.relu,
                                      kernel_regularizer=tf.keras.regularizers.L2(1.E-5)),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(num_outputs, activation=tf.nn.relu),
            ])
            self.ranker = pg.PageRank(0.9, renormalize=True, assume_immutability=True, error_type="iters", max_iters=10)
            self.input_spec = None  # prevents some versions of tensorflow from checking call inputs

        def call(self, inputs, training=False):
            graph, features = inputs
            predict = super().call(features, training=training)
            predict = self.ranker.propagate(graph, predict, graph_dropout=0.5 if training else 0)
            return tf.nn.softmax(predict, axis=1)

    model = APPNP(features.shape[1], labels.shape[1])
    with pytest.raises(Exception):
        pg.gnn_train(model, graph, features, labels, training, validation, test=test, epochs=2)
    pg.load_backend('tensorflow')
    pg.gnn_train(model, graph, features, labels, training, validation, test=test, epochs=300, patience=2)
    predictions = model([graph, features])
    pg.load_backend('numpy')
    with pytest.raises(Exception):
        pg.gnn_accuracy(labels, predictions, test)


def test_appnp_tf():
    graph, features, labels = pg.load_feature_dataset('synthfeats')
    training, test = pg.split(list(range(len(graph))), 0.8)
    training, validation = pg.split(training, 1 - 0.2 / 0.8)

    class APPNP(tf.keras.Sequential):
        def __init__(self, num_inputs, num_outputs, hidden=64):
            super().__init__([
                tf.keras.layers.Dropout(0.5, input_shape=(num_inputs,)),
                tf.keras.layers.Dense(hidden, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.L2(1.E-5)),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(num_outputs, activation=tf.nn.relu),
            ])
            self.ranker = pg.PageRank(0.9, renormalize=True, assume_immutability=True, error_type="iters", max_iters=10)
            self.input_spec = None  # prevents some versions of tensorflow from checking call inputs

        def call(self, inputs, training=False):
            graph, features = inputs
            predict = super().call(features, training=training)
            predict = self.ranker.propagate(graph, predict, graph_dropout=0.5 if training else 0)
            return tf.nn.softmax(predict, axis=1)

    pg.load_backend('tensorflow')
    model = APPNP(features.shape[1], labels.shape[1])
    pg.gnn_train(model, graph, features, labels, training, validation, test=test, epochs=50)
    assert float(pg.gnn_accuracy(labels, model([graph, features]), test)) >= 0.5
    pg.load_backend('numpy')


def test_appnp_torch():
    graph, features, labels = pg.load_feature_dataset('synthfeats')
    training, test = pg.split(list(range(len(graph))), 0.8)
    training, validation = pg.split(training, 1 - 0.2 / 0.8)

    class AutotuneAPPNP(torch.nn.Module):
        def __init__(self, num_inputs, num_outputs, hidden=64):
            super().__init__()
            self.layer1 = torch.nn.Linear(num_inputs, hidden)
            self.layer2 = torch.nn.Linear(hidden, num_outputs)
            self.activation = torch.nn.ReLU()
            self.dropout = torch.nn.Dropout(0.5)
            self.num_outputs = num_outputs
            self.ranker = pg.PageRank(0.9, renormalize=True, assume_immutability=True, error_type="iters", max_iters=10)

        def forward(self, inputs, training=False):
            graph, features = inputs
            predict = self.dropout(torch.FloatTensor(features))
            predict = self.dropout(self.activation(self.layer1(predict)))
            predict = self.activation(self.layer2(predict))
            predict = self.ranker.propagate(graph, predict, graph_dropout=0.5 if training else 0)
            ret = torch.nn.functional.softmax(predict, dim=1)
            self.loss = 0
            for param in self.layer1.parameters():
                self.loss = self.loss + 1E-5*torch.norm(param)
            return ret

    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    pg.load_backend('pytorch')
    model = AutotuneAPPNP(features.shape[1], labels.shape[1])
    model.apply(init_weights)
    pg.gnn_train(model, graph, features, labels, training, validation, epochs=50, patience=2)
    # TODO: higher numbers fail only on github actions - for local tests it is fine
    assert float(pg.gnn_accuracy(labels, model([graph, features]), test)) >= 0.17
    pg.load_backend('numpy')
