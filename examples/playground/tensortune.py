import tensorflow as tf
import pygrank as pg
from typing import Optional, Union



class CustomLoss(pg.Supervised):
    """Computes the L2 norm on the difference between scores and known scores."""

    def evaluate(self, scores: pg.GraphSignalData) -> pg.BackendPrimitive:
        known_scores, scores = self.to_numpy(scores)
        ret = pg.max(tf.nn.softmax((known_scores-scores)**2))
        return ret


class Tensortune(pg.Postprocessor):
    def __init__(self, ranker, pretrainer=None, model=None, postprocessor=pg.Tautology, gnn=True, robustness=0.05):
        self.ranker = ranker
        self.pretrainer = pretrainer
        self._model = model
        self.postprocessor = postprocessor
        self.robustness = robustness
        self.gnn = gnn

    def model(self):
        #if self._model is None:
        model = tf.keras.models.Sequential()
        model.add(tf.keras.Input(shape=(3,)))
        dims = 8

        class CustomSerializer(tf.keras.initializers.Initializer):
            def __call__(self, shape, dtype=None, **kwargs):
                return tf.concat([tf.constant(1, shape=(1, shape[1]),  dtype=dtype),
                                  tf.random.normal(shape=(shape[0]-1, shape[1]), dtype=dtype)*0.01], axis=0)

        model.add(tf.keras.layers.Dense(dims,
                                        kernel_initializer=CustomSerializer(),
                                        #kernel_regularizer=tf.keras.regularizers.L2(0.0005)
                                        ))
        for _ in range(2):
            model.add(tf.keras.layers.Dense(dims,
                                            kernel_initializer=CustomSerializer(),
                                            #kernel_regularizer=tf.keras.regularizers.L2(0.00001)
                                            ))
        model.add(tf.keras.layers.Dense(1,
                                        kernel_initializer=CustomSerializer(),
                                        #kernel_regularizer=tf.keras.regularizers.L2(0.00001),
                                        activation='relu'))
        self._model = model
        self._model.compile()
        return self._model

    def train_model(self, graph, personalization, sensitive, *args, **kwargs):
        original_personalization = personalization
        original_ranks = self.postprocessor(self.ranker)(graph, personalization, *args, **kwargs)
        prev_convergence = self.ranker.convergence
        self.ranker.convergence = pg.ConvergenceManager(error_type="iters", max_iters=prev_convergence.iteration)
        #self.ranker = pg.PageRank(0.9, error_type="iters", max_iters=10) # ablation study
        #pretrained_ranks = None if self.pretrainer is None else self.pretrainer(graph, personalization, *args, sensitive=sensitive, **kwargs)
        training_objective = pg.AM()\
            .add(pg.Mabs(tf.cast(original_ranks.np, tf.float32)), weight=1)\
            .add(pg.pRule(tf.cast(sensitive.np, tf.float32), exclude=tf.cast(personalization.np, tf.float32)),
                 max_val=0.8, weight=-10)
        model = self.model()
        with pg.Backend("tensorflow"):
            features = tf.concat([tf.reshape(personalization.np, (-1, 1)),
                                  tf.reshape(original_ranks.np, (-1, 1)),
                                  tf.reshape(sensitive.np, (-1, 1))
                                  ], axis=1)
            best_loss = float('inf')
            best_ranks = None
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)

            patience = 100
            repeats = 5
            best_repeat_loss = float('inf')
            for epoch in range(2000):
                with tf.GradientTape() as tape:
                    feats = model(features)
                    personalization = pg.to_signal(personalization, feats)
                    ranks = self.ranker(graph, personalization, *args, **kwargs)
                    loss = training_objective(ranks) + tf.reduce_sum(model.losses)
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                personalization = pg.to_signal(personalization, feats)
                if self.gnn:
                    ranks = self.ranker(graph, personalization, *args, **kwargs)
                else:
                    ranks = personalization
                validation_loss = training_objective(ranks)
                if validation_loss <= best_loss:
                    patience = 100
                    best_ranks = ranks
                    best_loss = validation_loss
                    print("\r"
                          "repeats left", repeats,
                          "epoch", epoch,
                          "deviation", float(pg.KLDivergence(tf.cast(original_ranks.np, tf.float32))(ranks)),
                          "prule", float(pg.pRule(tf.cast(sensitive.np, tf.float32),
                                                  exclude=tf.cast(original_personalization.np, tf.float32))(ranks)), end="")

                patience -= 1
                if patience == 0:
                    repeats -= 1
                    if repeats == 0 or best_loss >= best_repeat_loss:
                        break
                    best_repeat_loss = min(best_loss, best_repeat_loss)
                    patience = 100
                    best_loss = float('inf')
                    model = self.model()
                    features = tf.concat([tf.reshape(personalization.np, (-1, 1)),
                                          tf.reshape(original_ranks.np, (-1, 1)),
                                          tf.reshape(sensitive.np, (-1, 1))
                                          ], axis=1)
        print("\r", end="")

        #print("epoch", epoch,
        #      "deviation", float(pg.KLDivergence(tf.cast(original_ranks.np, tf.float32))(ranks)),
        #      "prule", float(pg.pRule(tf.cast(sensitive.np, tf.float32),
        #                              exclude=tf.cast(original_personalization.np, tf.float32))(ranks)))
        self.ranker.convergence = prev_convergence
        return best_ranks

    def rank(self, graph, personalization, sensitive, *args, **kwargs):
        personalization = pg.to_signal(graph, personalization)
        #if self.pretrainer is not None:
        #    pretrain_tuner = Tensortune(self.ranker, model=self.model())
        #    pretrain_tuner.train_model(graph, personalization, sensitive, *args, **kwargs)
        return self.train_model(graph, personalization, sensitive, *args, **kwargs)


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