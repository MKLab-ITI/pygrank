import pygrank as pg
import torch


class APPNP(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden=64, dropout=0.5):
        super().__init__()
        self.layer1 = torch.nn.Linear(num_inputs, hidden)
        self.layer2 = torch.nn.Linear(hidden, num_outputs)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)
        self.ranker = pg.PageRank(0.9, renormalize=True, assume_immutability=True, error_type="iters", max_iters=10)

    def forward(self, inputs):
        graph, features = inputs
        training = self.training
        predict = torch.FloatTensor(features)
        predict = self.dropout(self.activation(self.layer1(predict)))
        predict = self.activation(self.layer2(predict))
        predict = self.ranker.propagate(graph, predict, graph_dropout=0.5 if training else 0)
        ret = torch.nn.functional.softmax(predict, dim=1)
        self.loss = 0
        for param in self.layer1.parameters():
            self.loss = self.loss + 0.5E-4*torch.norm(param)
        return ret


graph, features, labels = pg.load_feature_dataset('synthfeats')
training, test = pg.split(list(range(len(graph))), 0.8)
training, validation = pg.split(training, 1 - 0.2 / 0.8)
pg.load_backend('pytorch')
model = APPNP(features.shape[1], labels.shape[1])
pg.gnn_train(model, graph, features, labels, training, validation, epochs=50)
