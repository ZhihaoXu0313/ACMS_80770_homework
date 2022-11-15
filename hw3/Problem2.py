"""
ACMS 80770-03: Deep Learning with Graphs
Instructor: Navid Shervani-Tabar
Fall 2022
University of Notre Dame
Homework 2: Programming assignment
Problem 2
"""
import torch

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from torch import nn
from torch.autograd.functional import jacobian

torch.manual_seed(0)


class GCN:
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
        At = A + torch.eye(200)
        num_neighbours = At.sum(dim=-1, keepdims=True)
        H = self.projection(H)
        H = torch.mm(At, H)
        H = H / num_neighbours
        return self.activate(H)


class MyModel(nn.Module):
    """
        model
    """

    def __init__(self, A):
        super(MyModel, self).__init__()
        # -- initialize layers
        self.gcn1 = GCN(200, 100)
        self.gcn2 = GCN(100, 50)
        self.gcn3 = GCN(50, 20)
        self.gcn4 = GCN(20, 20)
        self.gcn5 = GCN(20, 20)
        self.A = A

    def forward(self, h0):
        x = self.gcn1(self.A, h0)
        # x = self.gcn2(self.A, x)
        # x = self.gcn3(self.A, x)
        # x = self.gcn4(self.A, x)
        # x = self.gcn5(self.A, x)
        return x


"""
    Effective range
"""
# -- Initialize graph
seed = 32
n_V = 200  # total number of nodes
i = 27  # node ID
k = 1  # k-hop
G = nx.barabasi_albert_graph(n_V, 2, seed=seed)

# -- plot graph
layout = nx.spring_layout(G, seed=seed, iterations=400)
nx.draw(G, pos=layout, edge_color='gray', width=2, with_labels=False, node_size=100)

# -- plot neighborhood
nodes = nx.single_source_shortest_path_length(G, i, cutoff=k)
im2 = nx.draw_networkx_nodes(G, nodelist=nodes, label=nodes, pos=layout, node_color='red', node_size=100)

# -- visualize
plt.colorbar(im2)
plt.show()
plt.close()

"""
    Influence score
"""


# -- Initialize the model and node feature vectors

def one_hot(labels, Label_class):
    one_hot_label = np.array([[int(i == int(labels[j])) for i in range(Label_class)] for j in range(len(labels))])
    return one_hot_label


h0 = one_hot(list(nx.nodes(G)), 200)
A = nx.adjacency_matrix(G).todense()

model = MyModel(torch.Tensor(A))

H = model.forward(torch.Tensor(h0))
J = jacobian(model.forward, torch.Tensor(h0))


def Iscore(Jij):
    return abs(torch.sum(Jij))


# -- Influence sore
inf_score = [Iscore(J[i, :, j, :]).item() for j in range(200)]
print(inf_score)

# -- plot influence scores
layout = nx.spring_layout(G, seed=seed, iterations=400)
nx.draw(G, pos=layout, edge_color='gray', width=2, with_labels=False, node_size=100)
im3 = nx.draw_networkx_nodes(G, label=nodes, pos=layout, node_color=inf_score, node_size=100, cmap='Reds')
plt.colorbar(im3)
plt.show()
plt.close()
