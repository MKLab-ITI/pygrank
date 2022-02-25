import pygrank as pg
import tensorflow as tf


class Tensortune(pg.Postprocessor):
    def __init__(self, ranker):
        self.ranker = ranker

    def rank(self, graph, personalization, sensitive, *args, **kwargs):
        original_ranks = self.ranker(graph, personalization, *args, **kwargs)
        #training, validation = pg.split(personalization, 0.8)
        training, validation = None, None
        training_objective = pg.AM()\
            .add(pg.Mabs(original_ranks, exclude=validation), weight=-10.)\
            .add(pg.pRule(tf.cast(sensitive.np, tf.float32), exclude=validation), weight=1., max_val=.8)
        validation_objective = pg.AM()\
            .add(pg.Mabs(original_ranks, exclude=training), weight=-10.)\
            .add(pg.pRule(tf.cast(sensitive.np, tf.float32), exclude=training), weight=1., max_val=.8)

        model = tf.keras.models.Sequential()
        model.add(tf.keras.Input(shape=(3,)))
        model.add(tf.keras.layers.Dense(2, activation='tanh'))
        #model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        with pg.Backend("tensorflow"):
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
            features = tf.concat([tf.reshape(personalization.np, (-1, 1)),
                                  tf.reshape(original_ranks.np, (-1, 1)),
                                  tf.reshape(sensitive.np, (-1, 1))
                                  ], axis=1)
            best_loss = float('inf')
            best_ranks = None
            for epoch in range(2000):
                training_objective.weights = [-1., max(epoch/50.0, 1.)]
                with tf.GradientTape() as tape:
                    personalization = pg.to_signal(personalization, model(features))
                    ranks = self.ranker(graph, personalization, *args, **kwargs)
                    #ranks = tf.reshape(ranks.np, (1,-1))
                    ranks = pg.to_signal(ranks, ranks.np)
                    loss = -training_objective(ranks)
                    #for var in model.trainable_variables:
                    #    loss = loss + 1.E-5*tf.nn.l2_loss(tf.cast(var, tf.float64))
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                validation_loss = -validation_objective(ranks)
                if validation_loss < best_loss:
                    patience = 100
                    #print(epoch, 'best loss', float(validation_loss), float(loss))
                    best_ranks = ranks
                    best_loss = validation_loss
                patience -= 1
                if patience == 0:
                    break

        return best_ranks
