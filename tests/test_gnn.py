import pygrank as pg
import tensorflow as tf
import torch
import pytest


def test_gnn_errors():
    graph, features, labels = pg.load_feature_dataset('synthfeats')
    training, test = pg.split(list(range(len(graph))), 0.8)
    training, validation = pg.split(training, 1 - 0.2 / 0.8)

    from tensorflow.keras.layers import Dropout, Dense
    from tensorflow.keras.regularizers import L2

    class APPNP(tf.keras.Sequential):
        def __init__(self, num_inputs, num_outputs, hidden=64):
            super().__init__([
                Dropout(0.5, input_shape=(num_inputs,)),
                Dense(hidden, activation="relu", kernel_regularizer=L2(1.E-5)),
                Dropout(0.5),
                Dense(num_outputs, activation="relu")])
            self.ranker = pg.PageRank(0.9, renormalize=True, assume_immutability=True,
                                      use_quotient=False, error_type="iters", max_iters=10)  # 10 iterations

        def call(self, features, graph, training=False):
            predict = super().call(features, training=training)
            propagate = self.ranker.propagate(graph, predict, graph_dropout=0.5 if training else 0)
            return tf.nn.softmax(propagate, axis=1)

    model = APPNP(features.shape[1], labels.shape[1])
    with pytest.raises(Exception):
        pg.gnn_train(model, graph, features, labels, training, validation, test=test, epochs=2)
    pg.load_backend('tensorflow')
    pg.gnn_train(model, features, graph, labels, training, validation, test=test, epochs=300, patience=2)
    predictions = model(features, graph)
    pg.load_backend('numpy')
    with pytest.raises(Exception):
        pg.gnn_accuracy(labels, predictions, test)


def test_appnp_tf():
    from tensorflow.keras.layers import Dropout, Dense
    from tensorflow.keras.regularizers import L2

    class APPNP(tf.keras.Sequential):
        def __init__(self, num_inputs, num_outputs, hidden=64):
            super().__init__([
                Dropout(0.5, input_shape=(num_inputs,)),
                Dense(hidden, activation="relu", kernel_regularizer=L2(1.E-5)),
                Dropout(0.5),
                Dense(num_outputs, activation="relu")])
            self.ranker = pg.ParameterTuner(
                lambda par: pg.GenericGraphFilter([par[0] ** i for i in range(int(10))],
                                                  error_type="iters", max_iters=int(10)),
                max_vals=[0.95], min_vals=[0.5], verbose=False,
                measure=pg.Mabs, deviation_tol=0.1, tuning_backend="numpy")

        def call(self, features, graph, training=False):
            predict = super().call(features, training=training)
            propagate = self.ranker.propagate(graph, predict, graph_dropout=0.5 if training else 0)
            return tf.nn.softmax(propagate, axis=1)

    graph, features, labels = pg.load_feature_dataset('synthfeats')
    training, test = pg.split(list(range(len(graph))), 0.8)
    training, validation = pg.split(training, 1 - 0.2 / 0.8)
    model = APPNP(features.shape[1], labels.shape[1])
    with pg.Backend('tensorflow'):  # pygrank computations in tensorflow backend
        graph = pg.preprocessor(renormalize=True, cors=True)(graph)  # cors = use in many backends
        pg.gnn_train(model, features, graph, labels, training, validation,
                     optimizer=tf.optimizers.Adam(learning_rate=0.01), verbose=True, epochs=50)
        assert float(pg.gnn_accuracy(labels, model(features, graph), test)) == 1.  # dataset is super-easy to predict


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
            self.ranker = pg.ParameterTuner(
                lambda par: pg.GenericGraphFilter([par[0] ** i for i in range(int(10))],
                                                  error_type="iters", max_iters=int(10)),
                max_vals=[0.95], min_vals=[0.5], verbose=False,
                measure=pg.Mabs, deviation_tol=0.1, tuning_backend="numpy")

        def forward(self, features, graph, training=False):
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

    model = AutotuneAPPNP(features.shape[1], labels.shape[1])
    graph = pg.preprocessor(renormalize=True, cors=True)(graph)
    model.apply(init_weights)
    with pg.Backend('pytorch'):
        pg.gnn_train(model, features, graph, labels, training, validation, epochs=50)
        # TODO: investigate why this is not working as well as tf
        #assert float(pg.gnn_accuracy(labels, model(features, graph), test)) == 0.5
