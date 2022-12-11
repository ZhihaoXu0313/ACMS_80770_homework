import torch
import warnings
import numpy as np
import matplotlib.pyplot as plt

warnings.simplefilter(action='ignore', category=UserWarning)

from torch import nn
from rdkit import Chem
from torch.utils.data import Dataset
from torch.nn.functional import normalize
from sklearn.decomposition import PCA
from chainer_chemistry import datasets
from chainer_chemistry.dataset.preprocessors.ggnn_preprocessor import GGNNPreprocessor


class kernel:
    def __init__(self, K, R, d, J, lamb_max):
        # -- filter properties
        self.R = float(R)
        self.J = J
        self.K = K
        self.d = np.array(d)
        self.lamb_max = torch.tensor(lamb_max)

        # -- Half-Cosine kernel
        self.a = self.R * torch.log(self.lamb_max) / (self.J - self.R + 1)
        self.g_hat = lambda lamb: sum([self.d[k] * torch.cos(2 * np.pi * k * (lamb / self.a + 1 / 2))
                                       for k in range(self.K + 1)]) * (-lamb >= 0) * (-lamb <= self.a)

    def wavelet(self, lamb, j):
        """
        constructs wavelets ($j\in [2, J]$).
        :param lamb: eigenvalue (analogue of frequency).
        :param j: filter index in the filter bank.
        :return: filter response to input eigenvalues.
        """
        lamb[lamb < torch.exp(self.a / self.R * j - self.a)] = torch.exp(self.a / self.R * j - self.a)
        return self.g_hat(torch.log(lamb) - self.a / self.R * j)

    def scaling(self, lamb):
        """
        constructs scaling function (j=1).
        :param lamb: eigenvalue (analogue of frequency).
        :return: filter response to input eigenvalues.
        """
        norm_square = np.array([self.wavelet(lamb, i) ** 2 for i in range(1, self.J)])
        eps = self.R * self.d[0] ** 2 + 0.5 * self.R * sum(self.d[1:] ** 2) - sum(norm_square)
        eps[eps < 0] = 0
        return torch.sqrt(eps)


# -- load data
class MolecularDataset(Dataset):
    def __init__(self, N, train=True):
        if train:
            start, end = 0, 100000
        else:
            start, end = 100000, 130000

        dataset, dataset_smiles = datasets.get_qm9(GGNNPreprocessor(kekulize=True),
                                                   return_smiles=True,
                                                   target_index=np.random.choice(range(133000)[start:end], N, False))

        self.atom_types = [6, 8, 7, 9, 1]
        self.V = 9

        self.adjs = torch.stack(list(map(self.adj, dataset)))
        self.sigs = torch.stack(list(map(self.sig, dataset)))
        self.prop = torch.stack(list(map(self.target, dataset)))[:, 5]
        self.prop_2 = torch.stack(list(map(self.target_2, dataset_smiles)))

    def target_2(self, smiles):
        """
            compute the number of hydrogen-bond acceptor atoms
        :param smiles: smiles molecular representation
        :return:
        """
        mol = Chem.MolFromSmiles(smiles)

        return torch.tensor(Chem.rdMolDescriptors.CalcNumHBA(mol))

    def adj(self, x):
        x = x[1]
        adjacency = np.zeros((self.V, self.V)).astype(float)
        adjacency[:len(x[0]), :len(x[0])] = x[0] + 2 * x[1] + 3 * x[2]
        return torch.tensor(adjacency)

    def sig(self, x):
        x = x[0]
        atoms = np.ones((self.V)).astype(float)
        atoms[:len(x)] = x
        out = np.array([int(atom == atom_type) for atom_type in self.atom_types for atom in atoms]).astype(float)
        return torch.tensor(out).reshape(5, len(atoms)).T

    def target(self, x):
        """
            return Highest Occupied Molecular Orbital (HOMO) energy
        :param x:
        :return:
        """
        x = x[2]
        return torch.tensor(x)

    def __len__(self):
        return len(self.adjs)

    def __getitem__(self, item):
        return self.adjs[item], self.sigs[item], self.prop[item], self.prop_2[item]


class scattering(nn.Module):
    def __init__(self, J, L, V, d_f, K, d, R, lamb_max):
        super(scattering, self).__init__()

        # -- graph parameters
        self.n_node = V
        self.n_atom_features = d_f

        # -- filter parameters
        self.K = K
        self.d = d
        self.J = J
        self.R = R
        self.lamb_max = lamb_max
        self.filters = kernel(K=1, R=3, d=[0.5, 0.5], J=8, lamb_max=2)

        # -- scattering parameters
        self.L = L

    def compute_spectrum(self, W):
        """
            Computes eigenvalues of normalized graph Laplacian.
        :param W: tensor of graph adjacency matrices.
        :return: eigenvalues of normalized graph Laplacian
        """

        # -- computing Laplacian
        L = torch.diag_embed(W.sum(1)) - W

        # -- normalize Laplacian
        diag = W.sum(1)
        dhalf = torch.diag_embed(1. / torch.sqrt(torch.max(torch.ones(diag.size()), diag)))
        L = dhalf.matmul(L).matmul(dhalf)

        # -- eig decomposition
        E, V = torch.symeig(L, eigenvectors=True)

        return abs(E), V

    def filtering_matrices(self, W):
        """
            Compute filtering matrices (frames) for spectral filters
        :return: a collection of filtering matrices of each wavelet kernel and the scaling function in the filter-bank.
        """

        E, V = self.compute_spectrum(W)
        VT = torch.transpose(V, 2, 1)

        frame = torch.empty(V.shape[0], 0, self.n_node, self.n_node)

        # -- scaling frame
        filter_matrices = torch.diag_embed(self.filters.scaling(E))
        scaling_frame = V.matmul(filter_matrices).matmul(VT).unsqueeze(1)
        frame = torch.cat((frame, scaling_frame), dim=1)

        # -- wavelet frame
        for j in range(1, self.J):
            # -- calculate filters
            filter_matrices = torch.diag_embed(self.filters.wavelet(E, j))
            wavelet_frame = V.matmul(filter_matrices).matmul(VT).unsqueeze(1)
            frame = torch.cat((frame, wavelet_frame), dim=1)

        return frame

    def forward(self, W, f):
        """
            Perform wavelet scattering transform
        :param W: tensor of graph adjacency matrices.
        :param f: tensor of graph signal vectors.
        :return: wavelet scattering coefficients
        """

        # -- filtering matrices
        g = self.filtering_matrices(W)
        pooling = (1 / W.shape[1]) * torch.ones(W.shape[0], W.shape[1])

        # -- Initial U
        U = f.unsqueeze(1)

        # -- zero-th layer
        S = torch.bmm(f, pooling.unsqueeze(2).double())  # S_(0,1)
        pooling = pooling.unsqueeze(1).unsqueeze(3).repeat(1, self.J, 1, 1)

        for l in range(1, self.L + 1):
            layer = torch.empty([f.shape[0], 0, self.n_atom_features, self.n_node])

            for j in range(self.J ** (l - 1)):
                fj = U[:, j, :, :].unsqueeze(1)
                ghat = fj @ g

                layer_j = torch.abs(ghat)
                layer = torch.cat((layer, layer_j), dim=1)

                S_j = torch.transpose((layer_j @ pooling.double()).squeeze(3), 2, 1)
                S = torch.cat((S, S_j), dim=2)
            U = layer.clone()

        return S


# -- initialize scattering function
scat = scattering(L=2, V=9, d_f=5, K=1, R=3, d=[0.5, 0.5], J=8, lamb_max=2)

# -- load data
data = MolecularDataset(N=5000)
data_test = MolecularDataset(N=1000)

# -- Compute scattering feature maps
dims = 0
for l in range(scat.L + 1):
    dims += scat.J ** l

signal = data.sigs
signal_in = torch.transpose(signal.reshape(-1, scat.n_node, scat.n_atom_features), 2, 1)
scat_out = scat(data.adjs, signal_in).reshape(-1, dims * scat.n_atom_features)
print(scat_out.shape)
# -- PCA projection
pca = PCA(n_components=2)
latent_scat = pca.fit_transform(scat_out.detach().numpy())

# -- plot feature space
colormap = data.prop_2
plt.scatter(latent_scat[:, 0], latent_scat[:, 1], c=colormap)
plt.colorbar(label="number of hydrogen-bond acceptor atoms")
plt.show()

# -- NN training
from sklearn.neural_network import MLPRegressor


def MLPtrain(X_train, y_train, hidden_layer_sizes, max_iter):
    model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, random_state=42, max_iter=max_iter,
                         learning_rate_init=0.001, tol=1e-8,
                         activation='relu').fit(X_train, y_train)
    loss_values = model.loss_curve_

    return model, loss_values


hidden_layer_sizes = 100
max_iter = 500

model, loss_values = MLPtrain(scat_out, data.prop, hidden_layer_sizes, max_iter)

plt.plot(loss_values)
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()

signal_test = data_test.sigs
signal_in_test = torch.transpose(signal_test.reshape(-1, scat.n_node, scat.n_atom_features), 2, 1)
scat_out_test = scat(data_test.adjs, signal_in_test).reshape(-1, dims * scat.n_atom_features)
test_pred = model.predict(scat_out_test)

plt.scatter(test_pred, data_test.prop)
plt.plot([-0.3, -0.15], [-0.3, -0.15], c="black")
plt.xlim([-0.3, -0.15])
plt.ylim([-0.3, -0.15])
plt.xlabel("prediction")
plt.ylabel("target")
plt.show()
