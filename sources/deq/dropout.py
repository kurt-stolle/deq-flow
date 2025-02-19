import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter


class WeightDrop(torch.nn.Module):
    def __init__(self, module, weights, dropout=0, temporal=True):
        """
        Weight DropConnect, adapted from a recurrent setting by Merity et al. 2017

        :param module: The module whose weights are to be applied dropout on
        :param weights: A 2D list identifying the weights to be regularized. Each element of weights should be a
                        list containing the "path" to the weight kernel. For instance, if we want to regularize
                        module.layer2.weight3, then this should be ["layer2", "weight3"].
        :param dropout: The dropout rate (0 means no dropout)
        :param temporal: Whether we apply DropConnect only to the temporal parts of the weight (empirically we found
                         this not very important)
        """
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self.temporal = temporal
        if self.dropout > 0.0:
            self._setup()

    def _setup(self):
        for path in self.weights:
            full_name_w = ".".join(path)

            module = self.module
            name_w = path[-1]
            for i in range(len(path) - 1):
                module = getattr(module, path[i])
            w = getattr(module, name_w)
            del module._parameters[name_w]
            module.register_parameter(name_w + "_raw", Parameter(w.data))

    def _setweights(self):
        for path in self.weights:
            module = self.module
            name_w = path[-1]
            for i in range(len(path) - 1):
                module = getattr(module, path[i])
            raw_w = getattr(module, name_w + "_raw")

            if len(raw_w.size()) > 2 and raw_w.size(2) > 1 and self.temporal:
                # Drop the temporal parts of the weight; if 1x1 convolution then drop the whole kernel
                w = torch.cat(
                    [F.dropout(raw_w[:, :, :-1], p=self.dropout, training=self.training), raw_w[:, :, -1:]], dim=2
                )
            else:
                w = F.dropout(raw_w, p=self.dropout, training=self.training)

            setattr(module, name_w, w)

    def forward(self, *args, **kwargs):
        if self.dropout > 0.0:
            self._setweights()
        return self.module.forward(*args, **kwargs)


def matrix_diag(a, dim=2):
    """
    a has dimension (N, (L,) C), we want a matrix/batch diag that produces (N, (L,) C, C) from the last dimension of a
    """
    if dim == 2:
        res = torch.zeros(a.size(0), a.size(1), a.size(1))
        res.as_strided(a.size(), [res.stride(0), res.size(2) + 1]).copy_(a)
    else:
        res = torch.zeros(a.size(0), a.size(1), a.size(2), a.size(2))
        res.as_strided(a.size(), [res.stride(0), res.stride(1), res.size(3) + 1]).copy_(a)
    return res


##############################################################################################################
#
# Embedding dropout
#
##############################################################################################################


def embedded_dropout(embed, words, dropout=0.1, scale=None):
    """
    Apply embedding encoder (whose weight we apply a dropout)

    :param embed: The embedding layer
    :param words: The input sequence
    :param dropout: The embedding weight dropout rate
    :param scale: Scaling factor for the dropped embedding weight
    :return: The embedding output
    """
    if dropout:
        mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(
            embed.weight
        ) / (1 - dropout)
        mask = Variable(mask)
        masked_embed_weight = mask * embed.weight
    else:
        masked_embed_weight = embed.weight

    if scale:
        masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

    padding_idx = embed.padding_idx
    if padding_idx is None:
        padding_idx = -1

    X = F.embedding(
        words, masked_embed_weight, padding_idx, embed.max_norm, embed.norm_type, embed.scale_grad_by_freq, embed.sparse
    )
    return X


##############################################################################################################
#
# Variational dropout (for input/output layers, and for hidden layers)
#
##############################################################################################################


class VariationalHidDropout2d(nn.Module):
    def __init__(self, dropout=0.0):
        super(VariationalHidDropout2d, self).__init__()
        self.dropout = dropout
        self.mask = None

    def forward(self, x):
        if not self.training or self.dropout == 0:
            return x
        bsz, d, H, W = x.shape
        if self.mask is None:
            m = torch.zeros(bsz, d, H, W).bernoulli_(1 - self.dropout).to(x)
            self.mask = m.requires_grad_(False) / (1 - self.dropout)
        return self.mask * x
