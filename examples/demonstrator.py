import pygrank as pg
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense


class APPNP(tf.keras.Sequential):
    def __init__(self, num_inputs, num_outputs, hidden=64):
        super().__init__([
            Dropout(0.5, input_shape=(num_inputs,)),
            Dense(hidden, activation="relu", kernel_regularizer=tf.keras.regularizers.L2(1.E-5)),
            Dropout(0.5),
            Dense(num_outputs, activation="relu")])
        pre = pg.preprocessor(renormalize=True, assume_immutability=True)
        self.ranker = pg.ParameterTuner(
            lambda par: pg.GenericGraphFilter([par[0] ** i for i in range(int(par[1]))], preprocessor=pre,
                                              error_type="iters", max_iters=int(par[1])),
            max_vals=[0.95, 10], min_vals=[0.5, 5],
            measure=pg.Mabs, deviation_tol=0.1, tuning_backend="numpy")

    def call(self, inputs, training=False):
        graph, features = inputs
        predict = super().call(features, training=training)
        propagate = self.ranker.propagate(graph, predict, graph_dropout=0.5 if training else 0)
        return tf.nn.softmax(propagate, axis=1)


graph, features, labels = pg.load_feature_dataset('citeseer')
acc = 0
for _ in range(10):
    training, test = pg.split(list(range(len(graph))), 0.8)
    training, validation = pg.split(training, 1 - 0.2 / 0.8)
    pg.load_backend('tensorflow')  # explicitly load the appropriate backend
    model = APPNP(features.shape[1], labels.shape[1])
    pg.gnn_train(model, graph, features, labels, training, validation,
                 optimizer=tf.optimizers.Adam(learning_rate=0.01))
    acc += pg.gnn_accuracy(labels, model([graph, features]), test)/10
    print("Accuracy", pg.gnn_accuracy(labels, model([graph, features]), test))
print(acc)