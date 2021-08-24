import pygrank as pg
import tensorflow as tf


def test_appnp():
    graph, features, labels = pg.load_feature_dataset('synthfeats')
    training, test = pg.split(list(range(len(graph))), 0.8)
    training, validation = pg.split(training, 1 - 0.2 / 0.8)

    class AutotuneAPPNP:
        def __init__(self, num_inputs, num_outputs, hidden=64, dropout=0.5):
            self.mlp = tf.keras.Sequential([
                tf.keras.layers.Dropout(dropout, input_shape=(num_inputs,)),
                tf.keras.layers.Dense(hidden, activation=tf.nn.relu),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(num_outputs, activation=tf.nn.relu),
            ])
            self.num_outputs = num_outputs
            self.trainable_variables = self.mlp.trainable_variables
            pre = pg.preprocessor(renormalize=True, assume_immutability=True)
            self.ranker = pg.ParameterTuner(
                lambda params: pg.GenericGraphFilter([params[0]] * 10, preprocessor=pre, max_iters=10, error_type="iters"),
                max_vals=[0.95], min_vals=[0.5],
                measure=pg.KLDivergence, deviation_tol=0.1, tuning_backend="numpy", divide_range=2)

        def __call__(self, graph, features, training=False):
            predict = self.mlp(features, training=training)
            if not training:
                predict = self.ranker.propagate(graph, predict, graph_dropout=0.5 if training else 0)
            return tf.nn.softmax(predict, axis=1)

    pg.load_backend('tensorflow')
    model = AutotuneAPPNP(features.shape[1], labels.shape[1])
    pg.gnn_train(model, graph, features, labels, training, validation,
                 optimizer=tf.optimizers.Adam(learning_rate=0.01),
                 regularization=tf.keras.regularizers.L2(5.E-4),
                 epochs=50)
    assert float(pg.gnn_accuracy(labels, model(graph, features), test)) > 0.5
    pg.load_backend('numpy')
