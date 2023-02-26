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
        self.first_output = None

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


class IdentitySerializer(tf.keras.initializers.Initializer):
    def __call__(self, shape, dtype=None, **kwargs):
        return tf.concat([tf.constant(1, shape=(1, shape[1]),  dtype=dtype), tf.zeros(shape=(shape[0]-1, shape[1]), dtype=dtype)], axis=0)


class Tensortune(pg.Postprocessor):
    def __init__(self, ranker,
                 pretrainer=None,
                 model=None,
                 postprocessor=pg.Tautology,
                 fix_personalization=False,
                 gnn=True,
                 zero_mabs=0,
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
        self.dims = dims

    def model(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.Input(shape=(3,)))
        dims = self.dims

        for _ in range(self.depth):
            model.add(tf.keras.layers.Dense(dims, kernel_initializer=IdentitySerializer(), activation="relu"))
        model.add(tf.keras.layers.Dense(1, kernel_initializer=IdentitySerializer()))
        model.compile()
        return model

    def train_model(self, graph, personalization, sensitive, *args, **kwargs):
        original_ranks = self.postprocessor(self.ranker)(graph, personalization, *args, **kwargs)
        ordered = [original_ranks.get(v) for v in sorted(original_ranks, key=original_ranks.get, reverse=True)]
        self.min_diff = min([abs(ordered[i]-ordered[i+1]) for i in range(len(ordered)-1) if abs(ordered[i]-ordered[i+1]) > 0])
        prev_convergence = self.ranker.convergence
        self.ranker.convergence = pg.ConvergenceManager(error_type="iters", max_iters=prev_convergence.iteration)
        #self.ranker = pg.PageRank(0.9, error_type="iters", max_iters=10) # ablation study
        #pretrained_ranks = None if self.pretrainer is None else self.pretrainer(graph, personalization, *args, sensitive=sensitive, **kwargs)

        exclude = None if self.fix_personalization else tf.cast(personalization.np, tf.float32)
        if self.zero_mabs is None:
            training_objective = pg.AM(differentiable=False)\
                .add(pg.MSQRT(tf.cast(original_ranks.np, tf.float32), exclude=exclude), weight=1)\
                .add(pg.pRule(tf.cast(sensitive.np, tf.float32), exclude=exclude),
                     max_val=self.max_fairness, weight=-self.fairness_weight)
        else:
            training_objective = pg.AM(differentiable=True)\
                .add(pg.Mabs(tf.zeros(original_ranks.np.shape, tf.float32)), weight=float(self.zero_mabs))\
                .add(pg.Mabs(tf.cast(original_ranks.np, tf.float32),exclude),
                    weight=1)\
                .add(pg.pRule(tf.cast(sensitive.np, tf.float32), exclude),
                     max_val=self.max_fairness, weight=-self.fairness_weight)

        max_patience = 100
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
                    best_loss = validation_loss
                    if validation_loss <= best_repeat_loss:
                        best_repeat_loss = validation_loss
                        best_ranks = ranks
                    print("\r"
                          #"repeats left", repeats,
                          "epoch", epoch,
                          "depth", self.depth,
                          "mabs", float(pg.Mabs(tf.cast(original_ranks.np, tf.float32), exclude=exclude)(ranks)),
                          "prule", float(pg.pRule(tf.cast(sensitive.np, tf.float32), exclude=exclude)(ranks)), end="")

                patience -= 1
                if patience == 0:
                    break
                    #repeats -= 1
                    if best_loss > best_repeat_loss:
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
        return self.train_model(graph, personalization, sensitive, *args, **kwargs)
