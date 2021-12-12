import pygrank as pg
import tensorflow as tf


class APPNP:
    def __init__(self, num_inputs, num_outputs, hidden=64, dropout=0.5):
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dropout(dropout, input_shape=(num_inputs,)),
            tf.keras.layers.Dense(hidden, activation=tf.nn.relu),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(num_outputs, activation=tf.nn.relu),
        ])
        self.trainable_variables = self.mlp.trainable_variables
        pre = pg.preprocessor(renormalize=True, assume_immutability=True)
        self.ranker = pg.ParameterTuner(
            lambda params: pg.GenericGraphFilter([params[0]**i for i in range(int(params[1]))], preprocessor=pre, error_type="iters", max_iters=int(params[1])),
            max_vals=[0.95, 10], min_vals=[0.5, 5],
            measure=pg.Mabs, deviation_tol=0.1, tuning_backend="numpy")
        self.ranker = pg.PageRank(0.9, renormalize=True, assume_immutability=True, error_type="iters", max_iters=10)  # always 10 iterations

    def __call__(self, graph, features, training=False):
        predict = self.mlp(features, training=training)
        propagate = self.ranker.propagate(graph, predict)
        return tf.nn.softmax(propagate, axis=1)


graph, features, labels = pg.load_feature_dataset('citeseer')
training, test = pg.split(list(range(len(graph))), 0.8)
training, validation = pg.split(training, 1 - 0.2 / 0.8)
pg.load_backend('tensorflow')  # explicitly load the appropriate backend
model = APPNP(features.shape[1], labels.shape[1])
pg.gnn_train(model, graph, features, labels, training, validation,
             optimizer=tf.optimizers.Adam(learning_rate=0.01),
             regularization=tf.keras.regularizers.L2(5.E-4))
print("Accuracy", pg.gnn_accuracy(labels, model(graph, features), test))
