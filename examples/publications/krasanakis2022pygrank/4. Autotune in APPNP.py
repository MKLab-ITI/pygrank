import pygrank as pg
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.regularizers import L2


class APPNP(tf.keras.Sequential):
    def __init__(self, num_inputs, num_outputs, hidden=64):
        super().__init__([
            Dropout(0.5, input_shape=(num_inputs,)),
            Dense(hidden, activation="relu", kernel_regularizer=L2(0.005)),
            Dropout(0.5),
            Dense(num_outputs)])
        pre = pg.preprocessor(renormalize=True, assume_immutability=True)
        self.ranker = pg.ParameterTuner(
            lambda par: pg.GenericGraphFilter([par[0] ** i for i in range(int(10))],
                                              preprocessor=pre, error_type="iters", max_iters=10),
            max_vals=[1], min_vals=[0.5], verbose=False,
            measure=pg.Mabs, deviation_tol=0.01, tuning_backend="numpy")

    def call(self, features, graph, training=False):
        predict = super().call(features, training=training)
        propagate = self.ranker.propagate(graph, predict, graph_dropout=0.5*training)
        return tf.nn.softmax(propagate, axis=1)


graph, features, labels = pg.load_feature_dataset('citeseer')
training, test = pg.split(list(range(len(graph))), 0.8, seed=5)
training, validation = pg.split(training, 1 - 0.2 / 0.8)
model = APPNP(features.shape[1], labels.shape[1])
with pg.Backend('tensorflow'):  # pygrank computations in tensorflow backend
    pg.gnn_train(model, features, graph, labels, training, validation,
                 optimizer=tf.optimizers.Adam(learning_rate=0.01), verbose=True, test=test)
    print("Accuracy", pg.gnn_accuracy(labels, model(features, graph), test))
