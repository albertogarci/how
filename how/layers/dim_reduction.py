"""Layers implementing dimensionality reduction of a feature map"""

import torch
from torch import nn

from ..utils import whitening

class LinearDimReduction(nn.Linear):
    """Dimensionality reduction as a linear layer

    :param int input_dim: Network out_channels
    :param in dim: Whitening out_channels, for dimensionality reduction
    """

    def __init__(self, input_dim, dim):
        super().__init__(input_dim, dim, bias=True)

    def initialize_pca_whitening(self, des):
        """Initialize PCA whitening from given descriptors. Return tuple of shift and projection."""
        m, P = whitening.pcawhitenlearn_shrinkage(des)
        m, P = m.T, P.T

        projection = torch.Tensor(P[:self.weight.shape[0], :])
        self.weight.data = projection.to(self.weight.device)

        projected_shift = -torch.mm(torch.FloatTensor(P), torch.FloatTensor(m)).squeeze()
        self.bias.data = projected_shift[:self.weight.shape[0]].to(self.bias.device)
        return m, P

    def load_initialization(self, m, P):
        """Load PCA whitening from given shift and projection."""
        projection = torch.Tensor(P[:self.weight.shape[0], :])
        self.weight.data = projection.to(self.weight.device)

        projected_shift = -torch.mm(torch.FloatTensor(P), torch.FloatTensor(m)).squeeze()
        self.bias.data = projected_shift[:self.weight.shape[0]].to(self.bias.device)

class ConvDimReduction(nn.Conv2d):
    """Dimensionality reduction as a convolutional layer

    :param int input_dim: Network out_channels
    :param in dim: Whitening out_channels, for dimensionality reduction
    """

    def __init__(self, input_dim, dim):
        super().__init__(input_dim, dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)

    def initialize_pca_whitening(self, des):
        """Initialize PCA whitening from given descriptors. Return tuple of shift and projection."""
        m, P = whitening.pcawhitenlearn_shrinkage(des)
        m, P = m.T, P.T

        projection = torch.Tensor(P[:self.weight.shape[0], :]).unsqueeze(-1).unsqueeze(-1)
        self.weight.data = projection.to(self.weight.device)

        projected_shift = -torch.mm(torch.FloatTensor(P), torch.FloatTensor(m)).squeeze()
        self.bias.data = projected_shift[:self.weight.shape[0]].to(self.bias.device)
        return m, P

    def load_initialization(self, m, P):
        """Load PCA whitening from given shift and projection."""
        projection = torch.Tensor(P[:self.weight.shape[0], :]).unsqueeze(-1).unsqueeze(-1)
        self.weight.data = projection.to(self.weight.device)

        projected_shift = -torch.mm(torch.FloatTensor(P), torch.FloatTensor(m)).squeeze()
        self.bias.data = projected_shift[:self.weight.shape[0]].to(self.bias.device)


class ConvDimReduction3D(nn.Conv3d):
    """Dimensionality reduction as a convolutional layer

    :param int input_dim: Network out_channels
    :param in dim: Whitening out_channels, for dimensionality reduction
    """

    def __init__(self, input_dim, dim):
        super().__init__(input_dim, dim, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True)

    def initialize_pca_whitening(self, des):
        """Initialize PCA whitening from given descriptors. Return tuple of shift and projection."""
        m, P = whitening.pcawhitenlearn_shrinkage(des)
        m, P = m.T, P.T

        projection = torch.Tensor(P[:self.weight.shape[0], :]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        self.weight.data = projection.to(self.weight.device)

        projected_shift = -torch.mm(torch.FloatTensor(P), torch.FloatTensor(m)).squeeze()
        self.bias.data = projected_shift[:self.weight.shape[0]].to(self.bias.device)
        return m, P

    def load_initialization(self, m, P):
        """Load PCA whitening from given shift and projection."""
        projection = torch.Tensor(P[:self.weight.shape[0], :]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        self.weight.data = projection.to(self.weight.device)

        projected_shift = -torch.mm(torch.FloatTensor(P), torch.FloatTensor(m)).squeeze()
        self.bias.data = projected_shift[:self.weight.shape[0]].to(self.bias.device)
