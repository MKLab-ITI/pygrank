def gnn_accuracy(labels, predictions, nodes):
    import tensorflow as tf
    return float(1 - tf.math.count_nonzero(tf.argmax(tf.gather(labels, nodes, axis=0), axis=1) - tf.argmax(tf.gather(predictions, nodes, axis=0), axis=1)) / len(nodes))


def gnn_cross_entropy(labels, predictions, nodes):
    import tensorflow as tf
    return tf.keras.losses.CategoricalCrossentropy()(tf.gather(labels, nodes, axis=0), tf.gather(predictions, nodes, axis=0))


def gnn_train(model, graph, features, labels, training, validation,
              optimizer=None,
              regularization=None,
              patience=100,
              epochs=2000,
              test=None,
              verbose=False):
    import tensorflow as tf
    if optimizer is None:
        optimizer = tf.optimizers.Adam(learning_rate=0.01)
    if regularization is None:
        regularization = tf.keras.regularizers.L2(5.E-4)
    best_loss = float('inf')
    best_params = None
    if test is None:
        test = validation
    remaining_patience = patience
    regularized_variable = model.regularization if hasattr(model, "regularization") else model.trainable_variables
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            predictions = model(graph, features, training=True)
            loss = gnn_cross_entropy(labels, predictions, training)
            for param in regularized_variable:
                loss = loss + regularization(param)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        predictions = model(graph, features, training=False)
        loss = gnn_cross_entropy(labels, predictions, validation)
        remaining_patience -= 1
        if loss < best_loss:
            remaining_patience = patience
            best_loss = loss
            best_params = [tf.identity(param) for param in model.trainable_variables]
            if verbose:
                print("Epoch", epoch, "loss", float(loss), "acc", gnn_accuracy(labels, predictions, test))
        if remaining_patience == 0:
            print("Patience run out at epoch", epoch)
            break
    for variable, best_value in zip(model.trainable_variables, best_params):
        variable.assign(best_value)