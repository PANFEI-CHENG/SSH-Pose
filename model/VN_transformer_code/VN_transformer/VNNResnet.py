import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-6

class VNMaxPool(nn.Module):
    def __init__(self, in_channels):
        super(VNMaxPool, self).__init__()
        self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        d = self.map_to_dir(x.transpose(1,-1)).transpose(1,-1)
        dotprod = (x*d).sum(2, keepdims=True)
        idx = dotprod.max(dim=-1, keepdim=False)[1]
        index_tuple = torch.meshgrid([torch.arange(j) for j in x.size()[:-1]]) + (idx,)
        x_max = x[index_tuple]
        return x_max

def mean_pool(x, dim=-1, keepdim=False):
    return x.mean(dim=dim, keepdim=keepdim)

class VNAttenPool(nn.Module):
    def __init__(self, in_channels):
        super(VNAttenPool, self).__init__()
        self.map_to_dir = nn.Linear(in_channels, 1, bias=False)

    def forward(self, x, dim=-1, keepdim=False):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        w = (x * d).sum(2, keepdims=True)
        output = w * x
        return output.mean(dim=dim, keepdim=keepdim)

class VNBatchNorm(nn.Module):
    def __init__(self, num_features, dim):
        super(VNBatchNorm, self).__init__()
        self.num_features = num_features
        self.dim = dim
        if dim == 3 or dim == 4:
            self.bn = nn.BatchNorm1d(num_features)
        elif dim == 5:
            self.bn = nn.BatchNorm2d(num_features)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # norm = torch.sqrt((x*x).sum(2))
        if self.num_features != 1:
            norm = torch.norm(x, dim=2) + EPS
            norm_bn = self.bn(norm)
            norm = norm.unsqueeze(2)
            norm_bn = norm_bn.unsqueeze(2)
            x = x / norm * norm_bn
        
        return x

class VNLinearLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, negative_slope=0.2):
        super(VNLinearLeakyReLU, self).__init__()
        self.dim = dim
        self.negative_slope = negative_slope
        
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)
        self.batchnorm = VNBatchNorm(out_channels, dim=dim)

        
        if share_nonlinearity == True:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, out_channels, bias=False)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # Linear
        p = self.map_to_feat(x.transpose(1,-1)).transpose(1,-1)
        # BatchNorm
        p = self.batchnorm(p)
        # LeakyReLU
        d = self.map_to_dir(x.transpose(1,-1)).transpose(1,-1)
        dotprod = (p*d).sum(2, keepdims=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d*d).sum(2, keepdims=True)
        x_out = self.negative_slope * p + (1-self.negative_slope) * (mask*p + (1-mask)*(p-(dotprod/(d_norm_sq+EPS))*d))
        return x_out
def gather(x, idx, method=2):
    """
    implementation of a custom gather operation for faster backwards.
    :param x: input with shape [N, D_1, ... D_d]
    :param idx: indexing with shape [n_1, ..., n_m]
    :param method: Choice of the method
    :return: x[idx] with shape [n_1, ..., n_m, D_1, ... D_d]
    """

    if method == 0:
        return x[idx]
    elif method == 1:
        x = x.unsqueeze(1)
        x = x.expand((-1, idx.shape[-1], -1))
        idx = idx.unsqueeze(2)
        idx = idx.expand((-1, -1, x.shape[-1]))
        return x.gather(0, idx)
    elif method == 2:
        for i, ni in enumerate(idx.size()[1:]):
            x = x.unsqueeze(i + 1)
            new_s = list(x.size())
            new_s[i + 1] = ni
            x = x.expand(new_s)
        n = len(idx.size())
        for i, di in enumerate(x.size()[n:]):
            idx = idx.unsqueeze(i + n)
            new_s = list(idx.size())
            new_s[i + n] = di
            idx = idx.expand(new_s)
        return x.gather(0, idx)
    else:
        raise ValueError('Unkown method')

def max_pool(x, inds):
    """
    Pools features with the maximum values.
    :param x: [n1, d] features matrix
    :param inds: [n2, max_num] pooling indices
    :return: [n2, d] pooled features matrix
    """

    # Add a last row with minimum features for shadow pools
    # x = torch.cat((x, torch.zeros_like(x[:1, :])), 0)

    # Get all features for each pooling location [n2, max_num, d]
    b = x.shape[0]
    pool_features = x[torch.arange(b)[:,None,None], inds,:,:]

    # Pool the maximum [n2, d]
    max_features, _ = torch.max(pool_features, -3)
    return max_features
    
class VNNResnetBlock(nn.Module):

    def __init__(self, block_name, in_dim, out_dim, radius, scale, layer_ind, pooling='mean', mode='0'):
        """
        Initialize the first VNN block with its ReLU and BatchNorm.
        :param in_dim: dimension input features
        :param out_dim: dimension input features
        :param radius: current radius of convolution
        :param mode: '0' -- feature,
                     '1' -- feature, xyz,
                     '2' -- feature, xyz, mean,
                     '3' -- xyz, mean, feature
                     '4' -- xyz, mean, proj_xyz
                     '5' -- xyz, mean, proj_xyz, feature
        """
        super(VNNResnetBlock, self).__init__()

        # Get other parameters
        self.radius = radius
        self.scale = scale
        self.layer_ind = layer_ind
        self.block_name = block_name
        self.mode = mode

        if pooling == 'max':
            self.pool = VNMaxPool(out_dim // 2)
        elif pooling == 'mean':
            self.pool = mean_pool
        elif pooling == 'atten':
            self.pool = VNAttenPool(out_dim // 2)

        # self.pool2 = VNMaxPool(in_dim)
        if mode == '0':
            in_dim_ = in_dim
        elif mode == '1':
            in_dim_ = in_dim + 1
        elif mode == '2' or mode == '3' or mode == '5':
            in_dim_ = in_dim + 2
        elif mode == '4' or mode == '6':
            in_dim_ = in_dim + 3
        elif mode == '7':
            in_dim_ = in_dim + 4

        self.conv = VNLinearLeakyReLU(in_dim_, out_dim // 2)
        self.unary = VNLinearLeakyReLU(out_dim // 2, out_dim, dim=4)

        self.unary_shortcut = VNLinearLeakyReLU(in_dim, out_dim, dim=4)
        return

    def forward(self, features, batch):

        if 'strided' in self.block_name:
            q_pts = batch['points'][self.layer_ind + 1]
            s_pts = batch['points'][self.layer_ind]
            neighb_inds = batch['pools'][self.layer_ind]
        else:
            q_pts = batch['points'][self.layer_ind]
            s_pts = batch['points'][self.layer_ind]
            neighb_inds = batch['neighbors'][self.layer_ind]

        b, N, K = neighb_inds.shape

        # Add a fake point in the last row for shadow neighbors
        # s_pts = torch.cat((s_pts, torch.zeros_like(s_pts[:1, :]) + 1e6), 0)

        # Get neighbor points [n_points, n_neighbors, dim]
        neighbors = s_pts[torch.arange(b)[:,None,None], neighb_inds,:]

        # Replace the fake points by the corresponding query points
        # mask = (neighbors == 1e6)
        # neighbors = mask * q_pts[:, None] + neighbors * (~mask)

        # Center every neighborhood
        eqv_neighbors = neighbors - q_pts.unsqueeze(-2)

        #########################
        # scale normalization
        eqv_neighbors = eqv_neighbors / self.scale

        if self.mode == '0' and min(features.shape) == 0:
            raise ValueError('Features can not be empty!')

        # Add a zero feature for shadow neighbors
        # x = torch.cat((features, torch.zeros_like(features[:1, :])), 0)
        x = features.reshape(b, features.shape[1], -1)

        # Get the features of each neighborhood [n_points, n_neighbors, in_fdim]
        # neighb_x = gather(x, neighb_inds)
        neighb_x = x[torch.arange(b)[:,None,None], neighb_inds,:]

        if self.mode == '0':
            input = neighb_x
        elif self.mode == '1':
            # concatenate
            input = torch.cat([neighb_x, eqv_neighbors], dim=-1)
        elif self.mode == '2':
            N, K, C = neighbors.shape
            # calculate mean
            mean = eqv_neighbors.mean(-2, keepdim=True).repeat([1, K, 1])
            mean_cor = q_pts.unsqueeze(1) - mean  # neighbors - q_pts.unsqueeze(1).mean(0, keepdim=True) #
            # concatenate
            input = torch.cat([neighb_x, eqv_neighbors, mean], dim=-1)
        elif self.mode == '3':
            # calculate projection
            d = torch.sum(eqv_neighbors ** 2, dim=-1, keepdim=True).sqrt()
            proj_xyz = self.radius / d * eqv_neighbors
            # replace the nan element by zero
            proj_xyz = torch.nan_to_num(proj_xyz)
            # concatenate
            input = torch.cat([neighb_x, eqv_neighbors, proj_xyz], dim=-1)
        elif self.mode == '4':
            N, K, C = neighbors.shape
            # calculate mean
            mean = eqv_neighbors.mean(-2, keepdim=True).repeat([1, K, 1])
            mean_cor = neighbors - mean
            # calculate projection
            d = torch.sum(eqv_neighbors ** 2, dim=-1, keepdim=True).sqrt()
            proj_xyz = self.radius / d * eqv_neighbors
            # replace the nan element by zero
            proj_xyz = torch.nan_to_num(proj_xyz)
            # concatenate
            input = torch.cat([neighb_x, eqv_neighbors, mean_cor, proj_xyz], dim=-1)
        elif self.mode == '5':
            cros = torch.cross(neighb_x, eqv_neighbors)
            input = torch.cat([neighb_x, eqv_neighbors, cros], dim=-1)
        elif self.mode == '6':
            cros = torch.cross(neighb_x, eqv_neighbors)
            N, K, C = neighbors.shape
            # calculate mean
            mean = eqv_neighbors.mean(-2, keepdim=True).repeat([1, K, 1])
            input = torch.cat([neighb_x, eqv_neighbors, cros, mean], dim=-1)
        elif self.mode == '7':
            cros = torch.cross(neighb_x, eqv_neighbors)
            N, K, C = neighbors.shape
            # calculate mean
            mean = eqv_neighbors.mean(-2, keepdim=True).repeat([1, K, 1])
            # calculate projection
            d = torch.sum(eqv_neighbors ** 2, dim=-1, keepdim=True).sqrt()
            proj_xyz = self.radius / d * eqv_neighbors
            # replace the nan element by zero
            proj_xyz = torch.nan_to_num(proj_xyz)
            input = torch.cat([neighb_x, eqv_neighbors, cros, mean, proj_xyz], dim=-1)

        input = input.permute(0, 3, 1, 2).view(b, -1, 3, N, K)

        # VN conv
        input = self.conv(input)

        # pooling
        x = self.pool(input)

        # Second upscaling mlp
        x = self.unary(x)

        # Shortcut
        if 'strided' in self.block_name:
            # shortcut = max_pool(features, neighb_inds)
            # shortcut = self.pool2(features[torch.arange(b)[:,None,None], neighb_inds,:,:].permute(0, 3, 4, 1, 2))
            shortcut = mean_pool(features[torch.arange(b)[:,None,None], neighb_inds,:,:].permute(0, 3, 4, 1, 2))
        else:
            shortcut = features
        # N, C = shortcut.shape
        shortcut = self.unary_shortcut(shortcut)

        output_features = x + shortcut
        output_features = output_features.permute(0, 3, 1, 2)

        return output_features