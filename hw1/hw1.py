"""
ACMS 80770-03: Deep Learning with Graphs
Instructor: Navid Shervani-Tabar
Fall 2022
University of Notre Dame

Homework 1: Programming assignment
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from networkx.algorithms import bipartite
from networkx.generators.random_graphs import erdos_renyi_graph
import copy

"""
Self code for computing Jaccard's similarity
"""

def Jaccard(G):
  A = nx.to_numpy_array(G)
  n = np.size(A, 1)
  S = np.zeros((n, n))
  A2 = A@A
  for i in range(n):
    for j in range(n):
      S[i][j] = A2[i][j]/(sum(A[i])+sum(A[j])-A2[i][j])
  
  return S


# -- Initialize graphs
seed = 30
G = nx.florentine_families_graph()
nodes = G.nodes()

layout = nx.spring_layout(G, seed=seed)

# -- compute jaccard's similarity
"""
    This example is using NetwrokX's native implementation to compute similarities.
    Write a code to compute Jaccard's similarity and replace with this function.
"""
S = Jaccard(G) # self function

# pred = nx.jaccard_coefficient(G)

# -- keep a copy of edges in the graph
old_edges = copy.deepcopy(G.edges())

# -- add new edges representing similarities.
new_edges, metric = [], []
iG = list(nodes).index('Ginori')
print(nx.to_numpy_array(G)@nx.to_numpy_array(G)[iG])
for v in list(nodes):
  if v != 'Ginori':
    iv = list(nodes).index(v)
    G.add_edge("Ginori", v)
    print(f"(Ginori, {v}) -> {S[iG][iv]:.8f}")
    new_edges.append(('Ginori', v))
    metric.append(S[iG][iv])

# -- plot Florentine Families graph
nx.draw_networkx_nodes(G, nodelist=nodes, label=nodes, pos=layout, node_size=600)
nx.draw_networkx_edges(G, edgelist=old_edges, pos=layout, edge_color='gray', width=4)

# -- plot edges representing similarity
"""
    This example is randomly plotting similarities between 8 pairs of nodes in the graph. 
    Identify the ”Ginori”
"""
ne = nx.draw_networkx_edges(G, edgelist=new_edges, pos=layout, edge_color=np.asarray(metric), width=4, alpha=0.7)
plt.colorbar(ne)
plt.axis('off')
plt.show()
