import pygrank as pg
import tensorflow as tf


class APPNP(tf.keras.Sequential):
    def __init__(self, num_inputs, num_outputs, hidden=64, alpha=0.9, propagate_on_training=True, graph_dropout=0):
        super().__init__([
            tf.keras.layers.Dropout(0.5, input_shape=(num_inputs,)),
            tf.keras.layers.Dense(hidden, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.L2(1.E-5)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_outputs, activation=tf.nn.relu),
        ])

        if isinstance(alpha, str) and alpha == "estimated":
            self.ranker = pg.HopTuner(renormalize=True, assume_immutability=True, measure=pg.Cos,
                                      tuning_backend="numpy", tunable_offset=None, num_parameters=10, tol=0, autoregression=5, error_type="iters", max_iters=10)
        else:
            if isinstance(alpha, tf.Variable):
                self.trainable_variables = self.trainable_variables + [alpha]
            self.ranker = pg.PageRank(alpha, renormalize=True, assume_immutability=True, error_type="iters", max_iters=10)

        self.graph_dropout = graph_dropout
        self.num_outputs = num_outputs
        self.propagate_on_training = propagate_on_training

    def call(self, inputs, training=False):
        graph, features = inputs
        predict = super().call(features, training=training)
        if not training or self.propagate_on_training:
            predict = self.ranker.propagate(graph, predict, graph_dropout=self.graph_dropout if training else 0)

        return tf.nn.softmax(predict, axis=1)


pg.load_backend('numpy')
graph, features, labels = pg.load_feature_dataset('cora')
for seed in range(10):
    training, test = pg.split(list(range(len(graph))), 0.8, seed=seed)
    training, validation = pg.split(training, 1-0.2/0.8, seed=seed)
    architectures = {"APPNP": APPNP(features.shape[1], labels.shape[1], alpha=0.9),
                     #"LAPPNP": APPNP(features.shape[1], labels.shape[1], alpha=tf.Variable([0.85])),
                     "APFNP": APPNP(features.shape[1], labels.shape[1], alpha="estimated")
                     }

    pg.load_backend('tensorflow')
    accs = dict()
    for architecture, model in architectures.items():
        pg.gnn_train(model, graph, features, labels, training, validation, test=test)
        accs[architecture] = float(pg.gnn_accuracy(labels, model([graph, features]), test))
        print("seed"+str(seed)+" & "+" & ".join(str(acc) for acc in accs.values()))
