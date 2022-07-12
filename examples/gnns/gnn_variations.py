import pygrank as pg
import tensorflow as tf

optimization = dict()
pre = pg.preprocessor(assume_immutability=True, normalization="symmetric", renormalize=True)
convergence = {"error_type": "iters", "max_iters": 10}
algorithms = {
    "ppr0.5": pg.PageRank(alpha=0.5, preprocessor=pre, **convergence),
    "ppr0.85": pg.PageRank(alpha=0.85, preprocessor=pre, **convergence),
    "ppr0.9": pg.PageRank(alpha=0.9, preprocessor=pre, **convergence),
    "ppr0.99": pg.PageRank(alpha=0.99, preprocessor=pre, **convergence),
    "hk2": pg.HeatKernel(t=2, preprocessor=pre, **convergence, optimization_dict=optimization),
    "hk3": pg.HeatKernel(t=3, preprocessor=pre, **convergence, optimization_dict=optimization),
    "hk5": pg.HeatKernel(t=5, preprocessor=pre, **convergence, optimization_dict=optimization),
    "hk7": pg.HeatKernel(t=7, preprocessor=pre, **convergence, optimization_dict=optimization),
}


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
        #self.ranker = pg.Sweep(self.ranker)
        self.ranker = pg.ParameterTuner( measure=pg.Mabs, tuning_backend="numpy")

        self.input_spec = None  # prevents some versions of tensorflow from checking call inputs

    def call(self, inputs, training=False):
        graph, features = inputs
        predict = super().call(features, training=training)
        predict = self.ranker.propagate(graph, predict, graph_dropout=0.5 if training else 0)
        return tf.nn.softmax(predict, axis=1)


graph, features, labels = pg.load_feature_dataset('citeseer')
training, test = pg.split(list(range(len(graph))), 0.8)
training, validation = pg.split(training, 1 - 0.2 / 0.8)
pg.load_backend('tensorflow')  # explicitly load the appropriate backend
model = APPNP(features.shape[1], labels.shape[1])
pg.gnn_train(model, graph, features, labels, training, validation,
             optimizer=tf.optimizers.Adam(learning_rate=0.01), verbose=True)
print("Accuracy", pg.gnn_accuracy(labels, model([graph, features]), test))