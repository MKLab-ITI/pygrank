import tensorflow as tf
import pygrank as pg
from typing import Optional, Union


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
                    #print("epoch", epoch, "loss", validation_loss, "prule", pg.pRule(tf.cast(sensitive.np, tf.float32))(ranks))
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