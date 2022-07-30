import pygrank as pg
import tensorflow as tf
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
        self.input_spec = None  # allow non-tensor inputs to call(...)

    def call(self, inputs, training=False):
        graph, features = inputs
        predict = super().call(features, training=training)
        propagate = self.ranker.propagate(graph, predict, graph_dropout=0.5 if training else 0)
        return tf.nn.softmax(propagate, axis=1)


graph, features, labels = pg.load_feature_dataset('citeseer')
training, test = pg.split(list(range(len(graph))), 0.8)
training, validation = pg.split(training, 1 - 0.2 / 0.8)
model = APPNP(features.shape[1], labels.shape[1])
with pg.Backend('tensorflow'):  # pygrank computations in tensorflow backend
    pg.gnn_train(model, graph, features, labels, training, validation,
                 optimizer=tf.optimizers.Adam(learning_rate=0.01), verbose=True)
    print("Accuracy", pg.gnn_accuracy(labels, model([graph, features]), test))
