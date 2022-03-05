import tensorflow as tf
import pygrank as pg
from typing import Optional, Union


class LFPR(pg.RecursiveGraphFilter):
    def __init__(self, alpha: float = 0.85, redistributor: Optional[Union[str,pg.NodeRanking]] = None, target_pRule=1, *args, **kwargs):
        self.alpha = alpha
        kwargs["preprocessor"] = pg.preprocessor(assume_immutability=False, normalization="none")
        self.redistributor = redistributor
        self.target_pRule = target_pRule
        super().__init__(*args, **kwargs)

    def _start(self, M, personalization, ranks, sensitive, *args, **kwargs):
        sensitive = pg.to_signal(ranks, sensitive)
        outR = pg.conv(sensitive.np, M)
        outB = pg.conv(1.-sensitive.np, M)
        phi = pg.sum(sensitive.np)/pg.length(sensitive.np)*self.target_pRule
        dR = pg.ones(len(sensitive.graph))*0
        dB = pg.ones(len(sensitive.graph))*0
        for v,u in zip(*M.nonzero()):
            if outR[u] < phi*(outR[u]+outB[u]):
                M[u,v] = (1-phi)/outB[u]
                dR[u] = phi-(1-phi)/outB[u]*outR[u] # TODO: move these redundant computations in a separate for
            elif outR[u] != 0:
                M[u,v] = phi/outR[u]
                dB[u] = (1-phi)-phi/outR[u]*outB[u]
            else: # sink node
                dR[u] = phi
                dB[u] = 1-phi
        personalization.np = pg.safe_div(sensitive.np*personalization.np, pg.sum(sensitive.np))*self.target_pRule \
                                                 + pg.safe_div(personalization.np*(1-sensitive.np), pg.sum(1-sensitive.np))
        personalization.np = pg.safe_div(personalization.np, pg.sum(personalization.np))
        L = sensitive.np
        if self.redistributor is None or self.redistributor == "uniform":
            original_ranks = 1
        elif self.redistributor == "original":
            original_ranks = pg.PageRank(alpha=self.alpha,
                                         preprocessor=pg.preprocessor(assume_immutability=False, normalization="col"),
                                         convergence=self.convergence)(personalization).np
        else:
            original_ranks = self.redistributor(personalization).np

        self.dR = dR
        self.dB = dB
        self.xR = original_ranks*L / pg.sum(original_ranks*L)
        self.xB = original_ranks*(1-L) / pg.sum(original_ranks*(1-L))
        super()._start(M, personalization, ranks, *args, **kwargs)

    def _formula(self, M, personalization, ranks, sensitive, *args, **kwargs):
        deltaR = pg.sum(self.dR*ranks)
        deltaB = pg.sum(self.dB*ranks)
        return (pg.conv(ranks, M) + deltaR*self.xR + deltaB*self.xB) * self.alpha + personalization * (1 - self.alpha)

    def _end(self, M, personalization, ranks, *args, **kwargs):
        del self.xR
        del self.xB
        del self.dR
        del self.dB
        super()._end(M, personalization, ranks, *args, **kwargs)


class Tensortune(pg.Postprocessor):
    def __init__(self, ranker, pretrainer=None, model=None):
        self.ranker = ranker
        self.pretrainer = pretrainer
        self._model = model

    def model(self):
        if self._model is None:
            model = tf.keras.models.Sequential()
            model.add(tf.keras.Input(shape=(3,)))
            model.add(tf.keras.layers.Dropout(0.5))
            model.add(tf.keras.layers.Dense(16, activation='tanh'))
            model.add(tf.keras.layers.Dropout(0.5))
            model.add(tf.keras.layers.Dense(16, activation='tanh'))
            model.add(tf.keras.layers.Dropout(0.5))
            model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
            self._model = model
        return self._model

    def train_model(self, graph, personalization, sensitive, *args, **kwargs):
        original_ranks = self.ranker(graph, personalization, *args, **kwargs)
        #pretrained_ranks = None if self.pretrainer is None else self.pretrainer(graph, personalization, *args, sensitive=sensitive, **kwargs)
        features = tf.concat([tf.reshape(personalization.np, (-1, 1)),
                              tf.reshape(original_ranks.np, (-1, 1)),
                              tf.reshape(sensitive.np, (-1, 1))
                              ], axis=1)
        training_objective = pg.AM()\
            .add(pg.L2(tf.cast(original_ranks.np, tf.float32)), weight=1.)\
            .add(pg.pRule(tf.cast(sensitive.np, tf.float32)), max_val=0.8, weight=-10.)
        model = self.model()
        with pg.Backend("tensorflow"):
            best_loss = float('inf')
            best_ranks = None
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

            for epoch in range(5000):
                with tf.GradientTape() as tape:
                    personalization = pg.to_signal(personalization, model(features))
                    personalization.np = tf.nn.relu(personalization.np*2-1)
                    ranks = self.ranker(graph, personalization, *args, **kwargs)
                    loss = training_objective(ranks)
                    for var in model.trainable_variables:
                        loss = loss + 1.E-5 * tf.reduce_sum(var * var)
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                validation_loss = training_objective(ranks)
                if validation_loss < best_loss:
                    patience = 100
                    best_ranks = ranks
                    best_loss = validation_loss
                    print("epoch", epoch, "loss", validation_loss, "prule", pg.pRule(tf.cast(sensitive.np, tf.float32))(ranks))
                patience -= 1
                if patience == 0:
                    break
        return best_ranks

    def rank(self, graph, personalization, sensitive, *args, **kwargs):
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
            .add(pg.L2(base_ranks), weight=-1.)\
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