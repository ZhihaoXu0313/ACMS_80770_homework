"""
ACMS 80770-03: Deep Learning with Graphs
Instructor: Navid Shervani-Tabar
Fall 2022
University of Notre Dame
Homework 2: Programming assignment
Problem 1
"""
import torch
from torch import nn
import warnings
import numpy as np

warnings.simplefilter(action='ignore', category=UserWarning)
from chainer_chemistry import datasets
from chainer_chemistry.dataset.preprocessors.ggnn_preprocessor import GGNNPreprocessor

"""
    load data
"""
dataset, dataset_smiles = datasets.get_qm9(GGNNPreprocessor(kekulize=True), return_smiles=True,
                                           target_index=np.random.choice(range(133000), 6000, False))

V = 9
atom_types = [6, 8, 7, 9, 1]


def adj(x):
    x = x[1]
    adjacency = np.zeros((V, V)).astype(float)
    adjacency[:len(x[0]), :len(x[0])] = x[0] + 2 * x[1] + 3 * x[2]
    return torch.tensor(adjacency)


def sig(x):
    x = x[0]
    atoms = np.ones((V)).astype(float)
    atoms[:len(x)] = x
    out = np.array([int(atom == atom_type) for atom_type in atom_types for atom in atoms]).astype(float)
    return torch.tensor(out).reshape(5, len(atoms)).T


def target(x):
    x = x[2]
    return torch.tensor(x)


adjs = torch.stack(list(map(adj, dataset)))
sigs = torch.stack(list(map(sig, dataset)))
prop = torch.stack(list(map(target, dataset)))[:, 5]

# split train set and test set
adjs_train = adjs[:5000]
sigs_train = sigs[:5000]
prop_train = prop[:5000]

adjs_test = adjs[5000:]
sigs_test = sigs[5000:]
prop_test = prop[5000:]


class GCN(nn.Module):
    """
        Graph convolutional layer
    """

    def __init__(self, in_features, out_features):
        # -- initialize weight
        super(GCN, self).__init__()
        self.projection = nn.Linear(in_features, out_features, bias=False)
        nn.init.normal_(self.projection.weight, 0, 1)
        # nn.init.constant_(self.projection.bias, 0)

        # -- non-linearity
        self.activate = nn.Sigmoid()

    def __call__(self, A, H):
        # -- GCN propagation rule
        At = A + torch.eye(9)
        num_neighbours = At.sum(dim=-1, keepdims=True)
        H = self.projection(H)
        H = torch.bmm(At, H)
        H = H / num_neighbours
        return self.activate(H)


class GraphPooling:
    """
        Graph pooling layer
    """

    def __init__(self):
        super(GraphPooling, self).__init__()

    def __call__(self, H):
        # -- multi-set pooling operator
        pooling = H.sum(dim=-2, keepdim=H.dim() == 2)
        return pooling


class MyModel(nn.Module):
    """
        Regression  model
    """

    def __init__(self):
        super(MyModel, self).__init__()
        # -- initialize layers
        self.gcn = GCN(5, 3)
        self.pl = GraphPooling()
        self.fc1 = nn.Linear(3, 10)
        self.fc2 = nn.Linear(10, 1)
        # self.fc = nn.Linear(3, 1)

    def forward(self, A, h0):
        x = self.gcn(A, h0)
        x = self.pl(x)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        # x = self.fc(x)
        return x


"""
    Train
"""
# -- Initialize the model, loss function, and the optimizer
model = MyModel()
MyLoss = nn.MSELoss()
MyOptimizer = torch.optim.SGD(model.parameters(), lr=0.00001)

loss_hist = []
# -- update parameters
for epoch in range(500):
    for i in range(125):
        # -- predict
        pred = model(adjs_train[i * 40:(i + 1) * 40].float(), sigs_train[i * 40:(i + 1) * 40].float())

        # -- loss
        loss = MyLoss(pred, prop_train[i * 40:(i + 1) * 40].float())

        # -- optimize
        loss.backward()
        MyOptimizer.step()
        MyOptimizer.zero_grad()

    loss_hist.append(loss.item())

# -- plot loss
import matplotlib.pyplot as plt

plt.plot(loss_hist)
plt.xlabel("epochs")
plt.ylabel("error")
plt.show()

# -- test model

pred_test = model.forward(adjs_test.float(), sigs_test.float())
test = prop_test.float()

plt.scatter(pred_test.detach().numpy(), test.detach().numpy())
plt.xlabel("model prediction")
plt.ylabel("target")
plt.xlim([-0.4, 0])
plt.ylim([-0.4, 0])
plt.show()

#############
# - view the data points
y_train_pred = model.forward(adjs_train.float(), sigs_train.float())
plt.scatter(range(len(prop_train)), prop_train.data, color="blue")
plt.scatter(range(len(y_train_pred)), y_train_pred.data, color="red")
plt.show()


