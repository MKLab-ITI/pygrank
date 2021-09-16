import pygrank as pg
import tensorflow as tf
import math


class APPNP:
    def __init__(self, num_inputs, num_outputs, hidden=64, alpha=0.9, propagate_on_training=True):
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dropout(0.5, input_shape=(num_inputs,)),
            tf.keras.layers.Dense(hidden, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_outputs, activation=tf.nn.relu),
        ])
        self.trainable_variables = self.mlp.trainable_variables
        self.regularization = [self.mlp.layers[0].weights]

        if isinstance(alpha, str) and alpha == "estimated":
            self.ranker = pg.HopTuner(renormalize=True, assume_immutability=True, tuning_backend="numpy",
                                      measure=pg.Cos,#lambda *args: lambda x: pg.Cos(*args)(x),#math.exp(-pg.KLDivergence(*args)(x)),
                                      autoregression=0, error_type="iters",
                                      max_iters=20, num_parameters=20,
                                      tunable_offset=pg.Cos
                                      )
        elif isinstance(alpha, str) and alpha == "estimated_kl":
            self.ranker = pg.HopTuner(renormalize=True, assume_immutability=True, tuning_backend="numpy",
                                      measure=lambda *args: lambda x: (-pg.KLDivergence(*args)(x)),
                                      autoregression=0, error_type="iters",
                                      max_iters=10, num_parameters=10,
                                      tunable_offset=lambda *args: lambda x: (-pg.KLDivergence(*args)(x))
                                      )
        else:
            if isinstance(alpha, tf.Variable):
                self.trainable_variables = self.trainable_variables + [alpha]
            self.ranker = pg.PageRank(alpha, renormalize=True, assume_immutability=True, error_type="iters", max_iters=10)

        self.num_outputs = num_outputs
        self.propagate_on_training = propagate_on_training

    def __call__(self, graph, features, training=False):
        predict = self.mlp(features, training=training)
        if not training or self.propagate_on_training:
            predict = self.ranker.propagate(graph, predict, graph_dropout=0.5 if training else 0)
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
        #model.ranker.preprocessor.clear()
        pg.gnn_train(model, graph, features, labels, training, validation,
                     test=test, verbose=False, patience=100, epochs=2000)
        accs[architecture] = float(pg.gnn_accuracy(labels, model(graph, features), test))
        print(accs)
