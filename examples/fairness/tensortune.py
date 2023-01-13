import tensorflow as tf
import pygrank as pg
from typing import Optional, Union


class CustomLoss(pg.Supervised):
    """Computes the L2 norm on the difference between scores and known scores."""

    def evaluate(self, scores: pg.GraphSignalData) -> pg.BackendPrimitive:
        known_scores, scores = self.to_numpy(scores)
        ret = pg.log(pg.max(tf.nn.softmax((known_scores-scores)**2)))
        return ret


class Noise(tf.keras.layers.Layer):
    def __init__(self, error, **kwargs):
        self.supports_masking = True
        self.error = error
        #self.uses_learning_phase = True
        super(Noise, self).__init__(**kwargs)
        self.noise = None

    def call(self, x, mask=None):
        if x.shape[0] is None:
            return x
        if self.noise is None:
            self.noise = tf.random.uniform(shape=tf.keras.backend.shape(x), minval=1-self.error, maxval=1+self.error)
        #noise_x = x*self.noise
        #return tf.keras.backend.in_train_phase(noise_x, x)
        return x*self.noise

    def get_config(self):
        config = {'error': self.error}
        base_config = super(Noise, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Tensortune(pg.Postprocessor):
    def __init__(self, ranker,
                 pretrainer=None,
                 model=None,
                 postprocessor=pg.Tautology,
                 fix_personalization=False,
                 gnn=True,
                 zero_mabs=.01,
                 fairness_weight=1,
                 dims=5,
                 max_fairness=float('inf'),
                 robustness=0.0001):
        self.ranker = ranker
        self.pretrainer = pretrainer
        self._model = model
        self.postprocessor = postprocessor
        self.robustness = robustness
        self.gnn = gnn
        self.zero_mabs = zero_mabs
        self.fix_personalization = fix_personalization
        self.fairness_weight = fairness_weight
        self.max_fairness = max_fairness
        self._noise = None
        self.dims = dims

    def model(self):
        #if self._model is None:
        model = tf.keras.models.Sequential()
        model.add(tf.keras.Input(shape=(3,)))
        dims = self.dims

        class CustomSerializer(tf.keras.initializers.Initializer):
            def __call__(self, shape, dtype=None, **kwargs):
                return tf.concat([tf.constant(1, shape=(1, shape[1]),  dtype=dtype),
                                  tf.zeros(shape=(shape[0]-1, shape[1]), dtype=dtype)], axis=0)

        model.add(tf.keras.layers.Dense(dims,
                                        kernel_initializer=CustomSerializer(),
                                        activation="relu",
                                        #kernel_regularizer=tf.keras.regularizers.L2(0.0005)
                                        ))
        for _ in range(self.depth*5):
            model.add(tf.keras.layers.Dense(dims,
                                            kernel_initializer=CustomSerializer(),
                                            activation="relu",
                                            #kernel_regularizer=tf.keras.regularizers.L2(0.00001)
                                            ))

        model.add(tf.keras.layers.Dense(1,
                                        kernel_initializer=CustomSerializer(),
                                        #kernel_regularizer=tf.keras.regularizers.L2(0.00001)
                                        ))

        self._model = model
        self._model.compile()
        return self._model

    def train_model(self, graph, personalization, sensitive, *args, **kwargs):
        self._noise = None
        original_personalization = personalization
        original_ranks = self.postprocessor(self.ranker)(graph, personalization, *args, **kwargs)
        prev_convergence = self.ranker.convergence
        self.ranker.convergence = pg.ConvergenceManager(error_type="iters", max_iters=prev_convergence.iteration)
        #self.ranker = pg.PageRank(0.9, error_type="iters", max_iters=10) # ablation study
        #pretrained_ranks = None if self.pretrainer is None else self.pretrainer(graph, personalization, *args, sensitive=sensitive, **kwargs)

        if self.zero_mabs is None:
            training_objective = pg.AM(differentiable=False)\
                .add(pg.MSQRT(tf.cast(original_ranks.np, tf.float32)), weight=1)\
                .add(pg.pRule(tf.cast(sensitive.np, tf.float32), exclude=None if self.fix_personalization else tf.cast(personalization.np, tf.float32)),
                     max_val=self.max_fairness, weight=-self.fairness_weight)
        else:
            training_objective = pg.AM(differentiable=True)\
                .add(pg.Mabs(tf.zeros(original_ranks.np.shape, tf.float32)), weight=float(self.zero_mabs))\
                .add(pg.RMabs(tf.cast(original_ranks.np, tf.float32), exclude=None if self.fix_personalization else tf.cast(personalization.np, tf.float32)), weight=1)\
                .add(pg.pRule(tf.cast(sensitive.np, tf.float32), exclude=None if self.fix_personalization else tf.cast(personalization.np, tf.float32)),
                     max_val=self.max_fairness, weight=-self.fairness_weight)

        max_patience = 300
        self.depth = 1
        model = self.model()
        with pg.Backend("tensorflow"):
            features = tf.concat([tf.reshape(personalization.np, (-1, 1)),
                                  tf.reshape(original_ranks.np, (-1, 1)),
                                  tf.reshape(sensitive.np, (-1, 1))
                                  ], axis=1)

            best_loss = float('inf')
            best_ranks = None
            from examples.fairness.bounded_adam import AdamBounded
            optimizer = AdamBounded(learning_rate=0.001)

            patience = max_patience
            best_repeat_loss = float('inf')
            #noise = None
            for epoch in range(10000):
                with tf.GradientTape() as tape:
                    feats = model(features)
                    personalization = pg.to_signal(personalization, feats)
                    ranks = self.ranker(graph, personalization, *args, **kwargs)
                    #if noise is None:
                    #    noise = tf.random.uniform(minval=1-self.robustness, maxval=1+self.robustness, shape=ranks.np.shape)
                    #ranks = ranks*noise
                    loss = training_objective(ranks) #+ tf.reduce_sum(model.losses)
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                personalization = pg.to_signal(personalization, feats)
                if self.gnn:
                    ranks = self.ranker(graph, personalization, *args, **kwargs)
                else:
                    ranks = personalization
                #ranks = ranks*noise
                validation_loss = training_objective(ranks)
                if validation_loss <= best_loss:#-prev_convergence.tol:
                    patience = max_patience
                    if validation_loss <= best_repeat_loss:
                        best_ranks = ranks
                    best_loss = validation_loss
                    print("\r"
                          #"repeats left", repeats,
                          "epoch", epoch,
                          "depth", self.depth,
                          "mabs", float(pg.Mabs(tf.cast(original_ranks.np, tf.float32))(ranks)),
                          "prule", float(pg.pRule(tf.cast(sensitive.np, tf.float32),
                                                  exclude=None if self.fix_personalization else tf.cast(original_personalization.np, tf.float32))(ranks)), end="")

                patience -= 1
                if patience == 0:
                    #repeats -= 1
                    if best_loss >= best_repeat_loss:
                        break
                    best_repeat_loss = best_loss
                    patience = max_patience
                    self.depth += 1
                    model = self.model()
                    best_loss = float('inf')
                    #features = tf.concat([tf.reshape(personalization.np, (-1, 1)),
                    #                      tf.reshape(original_ranks.np, (-1, 1)),
                    #                      tf.reshape(sensitive.np, (-1, 1))
                    #                      ], axis=1)
        print("\r", end="")

        self.ranker.convergence = prev_convergence
        return best_ranks

    def rank(self, graph, personalization, sensitive, *args, **kwargs):
        personalization = pg.to_signal(graph, personalization)
        #if self.pretrainer is not None:
        #    pretrain_tuner = Tensortune(self.ranker, model=self.model())
        #    pretrain_tuner.train_model(graph, personalization, sensitive, *args, **kwargs)
        return self.train_model(graph, personalization, sensitive, *args, **kwargs)

"""
class TensortuneOutputs(pg.Postprocessor):
    def __init__(self, ranker, base_ranker=None):
        self.ranker = ranker
        self.base_ranker = ranker if base_ranker is None else base_ranker

    def rank(self, graph, personalization, sensitive, *args, **kwargs):
        original_ranks = self.ranker(graph, personalization, *args, sensitive=sensitive, **kwargs)
        base_ranks = original_ranks if self.ranker==self.base_ranker else self.base_ranker(graph, personalization, *args, **kwargs)
        training_objective = pg.AM()\
            .add(pg.KLDivergence(base_ranks), weight=-1.)\
            .add(pg.pRule(tf.cast(sensitive.np, tf.float32)), weight=10., max_val=0.8)

        with pg.Backend("tensorflow"):
            ranks_var = tf.Variable(pg.to_array(original_ranks.np))
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
            best_loss = float('inf')
            best_ranks = None
            for epoch in range(2000):
                with tf.GradientTape() as tape:
                    ranks = pg.to_signal(original_ranks, ranks_var)
                    loss = -training_objective(ranks) #+ 1.E-5*tf.reduce_sum(ranks_var*ranks_var)
                grads = tape.gradient(loss, [ranks_var])
                optimizer.apply_gradients(zip(grads, [ranks_var]))
                validation_loss = loss
                if validation_loss < best_loss:
                    patience = 100
                    best_ranks = ranks
                    best_loss = validation_loss
                patience -= 1
                if patience == 0:
                    break
        return best_ranks
"""
