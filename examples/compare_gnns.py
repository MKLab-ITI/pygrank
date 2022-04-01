import pygrank as pg
import tensorflow as tf


class APPNP(tf.keras.Sequential):
    def __init__(self, num_inputs, num_outputs, hidden=64, ranker=None):
        super().__init__([
            tf.keras.layers.Dropout(0.5, input_shape=(num_inputs,)),
            tf.keras.layers.Dense(hidden, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.L2(1.E-5)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_outputs, activation=tf.nn.relu),
        ])
        self.ranker = pg.PageRank(0.9, renormalize=True,
                                  assume_immutability=True,
                                  use_quotient=False,
                                  error_type="iters", max_iters=10) if ranker is None else ranker
        self.input_spec = None  # prevents some versions of tensorflow from checking call inputs

    def call(self, inputs, training=False):
        graph, features = inputs
        predict = super().call(features, training=training)
        predict = self.ranker.propagate(graph, predict, graph_dropout=0.5 if training else 0)
        return tf.nn.softmax(predict, axis=1)


pg.load_backend('numpy')
graph, features, labels = pg.load_feature_dataset('citeseer')
for seed in range(10):
    training, test = pg.split(list(range(len(graph))), 0.8, seed=seed)
    training, validation = pg.split(training, 1-0.2/0.8, seed=seed)
    architectures = {"APPNP": APPNP(features.shape[1], labels.shape[1]),
                     #"LAPPNP": APPNP(features.shape[1], labels.shape[1], alpha=tf.Variable([0.85])),
                     #"APFNP": APPNP(features.shape[1], labels.shape[1], alpha="estimated")
                     }

    with pg.Backend('tensorflow'):
        with tf.device('/GPU:1'):
            accs = dict()
            for architecture, model in architectures.items():
                pg.gnn_train(model, graph, features, labels, training, validation, test=test, verbose=True)
                accs[architecture] = float(pg.gnn_accuracy(labels, model([graph, features]), test))
                print("seed"+str(seed)+" & "+" & ".join(str(acc) for acc in accs.values()))
