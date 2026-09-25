import torch
import torch.nn.functional as F
from torch import nn, einsum, Tensor

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce

from model.VN_transformer_code.VN_transformer.attend import Attend
from model.VN_transformer_code.VN_transformer.VNNResnet import VNNResnetBlock


# helper

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def inner_dot_product(x, y, *, dim = -1, keepdim = True):
    return (x * y).sum(dim = dim, keepdim = keepdim)

# layernorm

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer('beta', torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

# equivariant modules

class VNLinear(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        bias_epsilon = 0.
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim_out, dim_in))

        self.bias = None
        self.bias_epsilon = bias_epsilon

        # in this paper, they propose going for quasi-equivariance with a small bias, controllable with epsilon, which they claim lead to better stability and results

        if bias_epsilon > 0.:
            self.bias = nn.Parameter(torch.randn(dim_out))

    def forward(self, x):
        out = einsum('... i c, o i -> ... o c', x, self.weight)

        if exists(self.bias):
            bias = F.normalize(self.bias, dim = -1) * self.bias_epsilon
            out = out + rearrange(bias, '... -> ... 1')

        return out

class VNReLU(nn.Module):
    def __init__(self, dim, eps = 1e-6):
        super().__init__()
        self.eps = eps
        self.W = nn.Parameter(torch.randn(dim, dim))
        self.U = nn.Parameter(torch.randn(dim, dim))

    def forward(self, x):
        q = einsum('... i c, o i -> ... o c', x, self.W)
        k = einsum('... i c, o i -> ... o c', x, self.U)

        qk = inner_dot_product(q, k)

        k_norm = k.norm(dim = -1, keepdim = True).clamp(min = self.eps)
        q_projected_on_k = q - inner_dot_product(q, k / k_norm) * k

        out = torch.where(
            qk >= 0.,
            q,
            q_projected_on_k
        )

        return out
    



class VNAttention_local(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        dim_coor = 3,
        bias_epsilon = 0.,
        l2_dist_attn = False,
        flash = False,
        num_latents = None   # setting this would enable perceiver-like cross attention from latents to sequence, with the latents derived from VNWeightedPool
    ):
        super().__init__()
        assert not (l2_dist_attn and flash), 'l2 distance attention is not compatible with flash attention'

        self.scale = (dim_coor * dim_head) ** -0.5
        dim_inner = dim_head * heads
        self.heads = heads

        self.to_q_input = None
        if exists(num_latents):
            self.to_q_input = VNWeightedPool(dim, num_pooled_tokens = num_latents, squeeze_out_pooled_dim = False)

        self.to_q = VNLinear(dim, dim_inner, bias_epsilon = bias_epsilon)
        self.to_k = VNLinear(dim, dim_inner, bias_epsilon = bias_epsilon)
        self.to_v = VNLinear(dim, dim_inner, bias_epsilon = bias_epsilon)
        self.to_out = VNLinear(dim_inner, dim, bias_epsilon = bias_epsilon)

        if l2_dist_attn and not exists(num_latents):
            # tied queries and keys for l2 distance attention, and not perceiver-like attention
            self.to_k = self.to_q

        self.attend = Attend(flash = flash, l2_dist = l2_dist_attn)

    def forward(self, x, node_index0, neibor_index0_1, featsq = None, mask = None):
        """
        einstein notation
        b - batch
        n - sequence
        h - heads
        d - feature dimension (channels)
        c - coordinate dimension (3 for 3d space)
        i - source sequence dimension
        j - target sequence dimension
        """
        b = x.shape[0]
        c = x.shape[-1]

        if exists(featsq):
            q = self.to_q(featsq)
        else:
            q = self.to_q(x)
            q = q[:,:node_index0,:,:]

        k, v = self.to_k(x), self.to_v(x)

         
        k = k[torch.arange(b)[:,None,None], neibor_index0_1,:,:]
        v = v[torch.arange(b)[:,None,None], neibor_index0_1,:,:]

        q = rearrange(q, '(b k) n (h d) c -> (b n) h k (d c)', b = b, h = self.heads)
        k, v = map(lambda t: rearrange(t, 'b n k (h d) c -> (b n) h k (d c)', h = self.heads), (k, v))

        out = self.attend(q, k, v, mask = mask)

        out = rearrange(out, '(b n) h k (d c) -> b (n k) (h d) c', b = b, c = c)
        return self.to_out(out)
    
class VNAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        dim_coor = 3,
        bias_epsilon = 0.,
        l2_dist_attn = False,
        flash = False,
        num_latents = None   # setting this would enable perceiver-like cross attention from latents to sequence, with the latents derived from VNWeightedPool
    ):
        super().__init__()
        assert not (l2_dist_attn and flash), 'l2 distance attention is not compatible with flash attention'

        self.scale = (dim_coor * dim_head) ** -0.5
        dim_inner = dim_head * heads
        self.heads = heads

        self.to_q_input = None
        if exists(num_latents):
            self.to_q_input = VNWeightedPool(dim, num_pooled_tokens = num_latents, squeeze_out_pooled_dim = False)

        self.to_q = VNLinear(dim, dim_inner, bias_epsilon = bias_epsilon)
        self.to_k = VNLinear(dim, dim_inner, bias_epsilon = bias_epsilon)
        self.to_v = VNLinear(dim, dim_inner, bias_epsilon = bias_epsilon)
        self.to_out = VNLinear(dim_inner, dim, bias_epsilon = bias_epsilon)

        if l2_dist_attn and not exists(num_latents):
            # tied queries and keys for l2 distance attention, and not perceiver-like attention
            self.to_k = self.to_q

        self.attend = Attend(flash = flash, l2_dist = l2_dist_attn)

    def forward(self, x, mask = None):
        """
        einstein notation
        b - batch
        n - sequence
        h - heads
        d - feature dimension (channels)
        c - coordinate dimension (3 for 3d space)
        i - source sequence dimension
        j - target sequence dimension
        """

        c = x.shape[-1]

        if exists(self.to_q_input):
            q_input = self.to_q_input(x, mask = mask)
        else:
            q_input = x

        q, k, v = self.to_q(q_input), self.to_k(x), self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) c -> b h n (d c)', h = self.heads), (q, k, v))

        out = self.attend(q, k, v, mask = mask)

        out = rearrange(out, 'b h n (d c) -> b n (h d) c', c = c)
        return self.to_out(out)

def VNFeedForward(dim, mult = 4, bias_epsilon = 0.):
    dim_inner = int(dim * mult)
    return nn.Sequential(
        VNLinear(dim, dim_inner, bias_epsilon = bias_epsilon),
        VNReLU(dim_inner),
        VNLinear(dim_inner, dim, bias_epsilon = bias_epsilon)
    )

class VNLayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-6):
        super().__init__()
        self.eps = eps
        self.ln = LayerNorm(dim)
        

    def forward(self, x):
        norms = x.norm(dim = -1)
        x = x / rearrange(norms.clamp(min = self.eps), '... -> ... 1')
        ln_out = self.ln(norms)
        return x * rearrange(ln_out, '... -> ... 1')

class VNWeightedPool(nn.Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        num_pooled_tokens = 1,
        squeeze_out_pooled_dim = True
    ):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.weight = nn.Parameter(torch.randn(num_pooled_tokens, dim, dim_out))
        self.squeeze_out_pooled_dim = num_pooled_tokens == 1 and squeeze_out_pooled_dim

    def forward(self, x, mask = None):
        if exists(mask):
            mask = rearrange(mask, 'b n -> b n 1 1')
            x = x.masked_fill(~mask, 0.)
            numer = reduce(x, 'b n d c -> b d c', 'sum')
            denom = mask.sum(dim = 1)
            mean_pooled = numer / denom.clamp(min = 1e-6)
        else:
            mean_pooled = reduce(x, 'b n d c -> b d c', 'mean')

        out = einsum('b d c, m d e -> b m e c', mean_pooled, self.weight)

        if not self.squeeze_out_pooled_dim:
            return out

        out = rearrange(out, 'b 1 d c -> b d c')
        return out

# equivariant VN transformer encoder

class VNTransformerEncoder_local(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        dim_head = 64,
        heads = 8,
        dim_coor = 3,
        ff_mult = 4,
        final_norm = False,
        bias_epsilon = 0.,
        l2_dist_attn = False,
        flash_attn = False
    ):
        super().__init__()
        self.dim = dim
        self.dim_coor = dim_coor

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                VNAttention_local(dim = dim, dim_head = dim_head, heads = heads, bias_epsilon = bias_epsilon, l2_dist_attn = l2_dist_attn, flash = flash_attn),
                VNLayerNorm(dim),
                VNFeedForward(dim = dim, mult = ff_mult, bias_epsilon = bias_epsilon),
                VNLayerNorm(dim)
            ]))

        self.norm = VNLayerNorm(dim) if final_norm else nn.Identity()

    def forward(
        self,
        x,
        node_index0, 
        neibor_index0_1,
        feats1 = None,
        mask = None
    ):
        *_, d, c = x.shape

        assert x.ndim == 4 and d == self.dim and c == self.dim_coor, 'input needs to be in the shape of (batch, seq, dim ({self.dim}), coordinate dim ({self.dim_coor}))'

        for attn, attn_post_ln, ff, ff_post_ln in self.layers:
            if node_index0 is not None:
                x = attn_post_ln(attn(x, node_index0, neibor_index0_1, mask = mask)) + x[:,:node_index0,:,:] 
                x = ff_post_ln(ff(x)) + x
            else:
                x = attn_post_ln(attn(x, node_index0, neibor_index0_1, feats1)) + feats1
                x = ff_post_ln(ff(x)) + x

        return self.norm(x)


class VNTransformerEncoder(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        dim_head = 64,
        heads = 8,
        dim_coor = 3,
        ff_mult = 4,
        final_norm = False,
        bias_epsilon = 0.,
        l2_dist_attn = False,
        flash_attn = False
    ):
        super().__init__()
        self.dim = dim
        self.dim_coor = dim_coor

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                VNAttention(dim = dim, dim_head = dim_head, heads = heads, bias_epsilon = bias_epsilon, l2_dist_attn = l2_dist_attn, flash = flash_attn),
                VNLayerNorm(dim),
                VNFeedForward(dim = dim, mult = ff_mult, bias_epsilon = bias_epsilon),
                VNLayerNorm(dim)
            ]))

        self.norm = VNLayerNorm(dim) if final_norm else nn.Identity()

    def forward(
        self,
        x,
        mask = None
    ):
        *_, d, c = x.shape

        assert x.ndim == 4 and d == self.dim and c == self.dim_coor, 'input needs to be in the shape of (batch, seq, dim ({self.dim}), coordinate dim ({self.dim_coor}))'

        for attn, attn_post_ln, ff, ff_post_ln in self.layers:
            x = attn_post_ln(attn(x, mask = mask)) + x
            x = ff_post_ln(ff(x)) + x

        return self.norm(x)
# invariant layers

class VNInvariant(nn.Module):
    def __init__(
        self,
        dim,
        dim_coor = 3,

    ):
        super().__init__()
        self.mlp = nn.Sequential(
            VNLinear(dim, dim_coor),
            VNReLU(dim_coor),
            Rearrange('... d e -> ... e d')
        )

    def forward(self, x):
        return einsum('b n d i, b n i o -> b n o', x, self.mlp(x))

# main class

class VNTransformer(nn.Module):
    def __init__(
        self,
        dim_in,
        dim,
        depth,
        num_tokens = None,
        dim_feat = None,
        dim_head = 64,
        heads = 8,
        dim_coor = 3,
        reduce_dim_out = False,
        bias_epsilon = 0.,
        l2_dist_attn = False,
        flash_attn = False,
        translation_equivariance = False,
        translation_invariant = False
    ):
        super().__init__()
        # self.token_emb = nn.Embedding(num_tokens, dim) if exists(num_tokens) else None

        dim_feat = default(dim_feat, 0)
        self.dim_feat = dim_feat
        self.dim_coor_total = dim_coor + dim_feat

        assert (int(translation_equivariance) + int(translation_invariant)) <= 1
        self.translation_equivariance = translation_equivariance
        self.translation_invariant = translation_invariant

        self.vn_proj_in = nn.Sequential(
            VNLinear(dim_in, dim, bias_epsilon = bias_epsilon)
        )

        self.encoder = VNTransformerEncoder(
            dim = dim,
            depth = depth,
            dim_head = dim_head,
            heads = heads,
            bias_epsilon = bias_epsilon,
            dim_coor = self.dim_coor_total,
            l2_dist_attn = l2_dist_attn,
            flash_attn = flash_attn
        )

        if reduce_dim_out:
            self.vn_proj_out = nn.Sequential(
                VNLayerNorm(dim),
                VNLinear(dim, 1, bias_epsilon = bias_epsilon),
                Rearrange('... 1 c -> ... c')
            )
        else:
            self.vn_proj_out = nn.Identity()

    def forward(
        self,
        ptsf, 
        normals, 
        feats_pts, 
        feats_rgb = None,
        mask = None,
        return_concatted_coors_and_feats = False
    ):

        x = feats_pts

 
        # x = torch.cat((x, feats_rgb), dim = -1)

        assert x.shape[-1] == self.dim_coor_total

        x = self.vn_proj_in(x)
        x = self.encoder(x, mask = mask)
        x = self.vn_proj_out(x)

        # coors_out, feats_out = x[..., :3], x[..., 3:]


        # if return_concatted_coors_and_feats:
        #     return torch.cat((coors_out, feats_out), dim = -1)

        # return coors_out, feats_out
        return x


class Local_VNTransformer(nn.Module):
    def __init__(
        self,
        *,
        dim_in,
        dim,
        depth,
        num_tokens = None,
        dim_feat = None,
        dim_head = 64,
        heads = 8,
        dim_coor = 3,
        reduce_dim_out = False,
        bias_epsilon = 0.,
        l2_dist_attn = False,
        flash_attn = False,
        translation_equivariance = False,
        translation_invariant = False
    ):
        super().__init__()
        # self.token_emb = nn.Embedding(num_tokens, dim) if exists(num_tokens) else None

        dim_feat = default(dim_feat, 0)
        self.dim_feat = dim_feat
        self.dim_coor_total = dim_coor + dim_feat

        assert (int(translation_equivariance) + int(translation_invariant)) <= 1
        self.translation_equivariance = translation_equivariance
        self.translation_invariant = translation_invariant

        self.vn_proj_in = nn.Sequential(
            # Rearrange('... c -> ... 1 c'),
            VNLinear(dim_in, dim, bias_epsilon = bias_epsilon)
        )

        self.encoder = VNTransformerEncoder_local(
            dim = dim,
            depth = depth,
            dim_head = dim_head,
            heads = heads,
            bias_epsilon = bias_epsilon,
            dim_coor = self.dim_coor_total,
            l2_dist_attn = l2_dist_attn,
            flash_attn = flash_attn
        )

        if reduce_dim_out:
            self.vn_proj_out = nn.Sequential(
                VNLayerNorm(dim),
                VNLinear(dim, 1, bias_epsilon = bias_epsilon),
                Rearrange('... 1 c -> ... c')
            )
        else:
            self.vn_proj_out = nn.Identity()

    def calc_Fourfeature(self, pts, normal, neighbor_idx):
        b,n,k = neighbor_idx.shape
        neighbor_pts = pts[torch.arange(b)[:,None,None], neighbor_idx]
        neighbor_normal = normal[torch.arange(b)[:,None,None], neighbor_idx]
        eqv_neighbors = neighbor_pts - pts.unsqueeze(-2)

        cros = torch.cross(neighbor_normal, eqv_neighbors)
        mean = eqv_neighbors.mean(-2, keepdim=True).repeat([1, 1, k, 1])
        feat0_0 = torch.cat([neighbor_normal, eqv_neighbors, cros, mean], dim=-1)
        feat0_0 = feat0_0.unsqueeze(-1).view(b, n, k, -1, 3)
        return feat0_0
    
    def forward(
        self,
        pts0,
        normals0,
        node_index0,
        neibor_index0_1,
        pts1 = None,
        rgb1 = None,
        normals1 = None,
        feats_pts0 = None,
        feats_rgb0 = None,
        feats_pts1 = None,
        feats_rgb1 = None,
        mask = None,  
        return_concatted_coors_and_feats = False
    ):
        if feats_pts0 is None:
            feats_pts0 = pts0

        if feats_rgb1 is not None:
            feats1 = torch.cat((feats_pts1, feats_rgb1), dim = -1)
        elif feats_pts1 is not None:
            feats1 = feats_pts1
            
        else:
            node_index0 = neibor_index0_1.shape[1]
            feats1 = None
        # if feats_pts is None:
        #     with torch.no_grad():
        #         feats_pts = self.calc_Fourfeature(pts, normals, neibor_index)
        # assert feats_rgb0.shape[-1] == self.dim_feat, f'dim_feat should be set to {feats_rgb0.shape[-1]}'
        # feats = torch.cat((feats_pts0, feats_rgb0), dim = -1)
        feats = feats_pts0

        assert feats.shape[-1] == self.dim_coor_total
        if len(feats.shape) == 3:
             feats = rearrange(feats, '... c -> ... 1 c')
        x = self.vn_proj_in(feats)
        x = self.encoder(x, node_index0, neibor_index0_1, feats1)
        x = self.vn_proj_out(x)



        return x
    

class VN_Trandown(nn.Module):
    def __init__(
        self,
        *,
        dim_in,
        dim,
        depth,
        num_tokens = None,
        dim_feat = None,
        dim_head = 64,
        heads = 8,
        dim_coor = 3,
        reduce_dim_out = True,
        bias_epsilon = 0.,
        l2_dist_attn = False,
        flash_attn = False,
        translation_equivariance = False,
        translation_invariant = False
    ):
        super().__init__()
        if dim_in == 64:
            self.VNNResnet = VNNResnetBlock(block_name='VNN_resnetb_strided', in_dim=dim_in, out_dim=dim, radius=None, scale=1.0, layer_ind=0, pooling='mean', mode='1')

        self.transformer0_1 = Local_VNTransformer(dim_in = dim_in, dim = dim, depth = 1, dim_head = dim_head, heads = heads, dim_feat = dim_feat, bias_epsilon = bias_epsilon)
        self.transformer1_1 = Local_VNTransformer(dim_in = dim, dim = dim, depth = depth-1, dim_head = dim_head, heads = heads, dim_feat = dim_feat, bias_epsilon = bias_epsilon)

    def forward(
        self,
        pts0,
        normals0,
        node_index0,
        neibor_index0_1,
        neibor_index1_1,
        pts1 = None,
        rgb1 = None,
        normals1 = None,
        feats_pts0 = None,
        feats_rgb0 = None,
        feats_pts1 = None,
        feats_rgb1 = None,
    ):
        
        if (feats_pts1 is None) != (feats_rgb1 is None) or node_index0 is None:
            batch = {'points':[pts0, pts1], 
                     'pools':[neibor_index0_1[:,:,:3]]}
            feats_pts1 = self.VNNResnet(feats_pts0, batch)
            # feats_rgb1 = feats_rgb1.view(feats_rgb1.shape[0],feats_rgb1.shape[1],-1,3)

        feats_pts1 = self.transformer0_1(pts0, normals0, node_index0, neibor_index0_1, 
                                                     pts1=pts1, rgb1=rgb1, normals1=normals1, 
                                                     feats_pts0=feats_pts0, feats_rgb0=feats_rgb0, 
                                                     feats_pts1 = feats_pts1, feats_rgb1 = feats_rgb1)
        feats_pts1 = self.transformer1_1(pts1, normals1, node_index0, neibor_index1_1, 
                                                     feats_pts0=feats_pts1, feats_rgb0=feats_rgb1)
        
        return feats_pts1
