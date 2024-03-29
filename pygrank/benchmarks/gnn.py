from pygrank.core import backend, utils


def _gnn_accuracy_tf(labels, predictions, nodes):
    import tensorflow as tf
    return float(1 - tf.math.count_nonzero(tf.argmax(tf.gather(labels, nodes, axis=0), axis=1) - tf.argmax(tf.gather(predictions, nodes, axis=0), axis=1)) / len(nodes))


def _gnn_cross_entropy_tf(labels, predictions, nodes):
    import tensorflow as tf
    return tf.keras.losses.CategoricalCrossentropy()(tf.gather(labels, nodes, axis=0), tf.gather(predictions, nodes, axis=0))


def _gnn_accuracy_torch(labels, predictions, nodes):
    import torch
    labels = torch.FloatTensor(labels)
    predictions = torch.FloatTensor(predictions)
    nodes = torch.LongTensor(nodes)
    return float(1 - torch.count_nonzero(torch.argmax(labels[nodes], dim=1) - torch.argmax(predictions[nodes], dim=1)) / len(nodes))


def _gnn_cross_entropy_torch(labels, predictions, nodes):
    import torch
    return torch.nn.BCELoss()(predictions[nodes], labels[nodes])


def gnn_train(*args, **kwargs):
    if backend.backend_name() == "tensorflow":
        return _gnn_train_tf(*args, **kwargs)
    elif backend.backend_name() == "pytorch":
        return _gnn_train_torch(*args, **kwargs)
    raise Exception("GNN training is supported only for tensorflow and pytorch backends")


def gnn_accuracy(labels, predictions, nodes):
    if backend.backend_name() == "tensorflow":
        return _gnn_accuracy_tf(labels, predictions, nodes)
    elif backend.backend_name() == "pytorch":
        return _gnn_accuracy_torch(labels, predictions, nodes)
    raise Exception("GNN accuracy is supported only for tensorflow and pytorch backends")


def _gnn_train_tf(model, features, graph, labels, training, validation,
              optimizer=None,
              patience=100,
              epochs=10000,
              test=None,
              verbose=False):
    import tensorflow as tf
    optimizer = tf.optimizers.Adam(learning_rate=0.01) if optimizer is None else optimizer
    best_loss = float('inf')
    best_params = None
    test = validation if test is None else test
    remaining_patience = patience
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            predictions = model(features, graph, training=True)
            loss = _gnn_cross_entropy_tf(labels, predictions, training)
            loss = loss + tf.reduce_sum(model.losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        predictions = model(features, graph, training=False)
        loss = _gnn_cross_entropy_tf(labels, predictions, validation)
        remaining_patience -= 1
        if loss < best_loss:
            remaining_patience = patience
            best_loss = loss
            best_params = [tf.identity(param) for param in model.trainable_variables]
            if verbose:   # pragma: no cover
                utils.log(f"Epoch {epoch} loss {loss} acc {float(_gnn_accuracy_tf(labels, predictions, test)):.3f}")
        if remaining_patience == 0:
            break
    if verbose:
        utils.log()
    for variable, best_value in zip(model.trainable_variables, best_params):
        variable.assign(best_value)


def _gnn_train_torch(model, features, graph, labels, training, validation,
              optimizer=None,
              patience=100,
              epochs=10000,
              test=None,
              verbose=False):
    import torch
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01) if optimizer is None else optimizer
    remaining_patience = patience
    test = validation if test is None else test
    labels = torch.FloatTensor(labels)
    features = torch.FloatTensor(features)
    training = torch.LongTensor(training)
    test = torch.LongTensor(test)
    validation = torch.LongTensor(validation)
    best_loss = float('inf')
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(features, graph, training=True)
        loss = _gnn_cross_entropy_torch(labels, predictions, training) + model.loss
        loss.backward()
        optimizer.step()
        loss = _gnn_cross_entropy_torch(labels, predictions, validation)
        remaining_patience -= 1
        if loss < best_loss:
            remaining_patience = patience
            best_loss = loss
            torch.save(model.state_dict(), "_pygrank_torch_state.pt")
            if verbose:   # pragma: no cover
                utils.log(f"Epoch {epoch} loss {loss} acc {float(_gnn_accuracy_torch(labels, predictions, test)):.3f}")
        if remaining_patience == 0:
            break
    utils.log()
    model.load_state_dict(torch.load("_pygrank_torch_state.pt"))
    model.eval()
    import os
    os.remove("_pygrank_torch_state.pt")
