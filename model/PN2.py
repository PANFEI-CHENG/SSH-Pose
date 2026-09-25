import torch
import torch.nn as nn
from module import PointNet2MSG
import torch.nn.functional as F
from extractor_dino import ViTExtractor
from torchvision import transforms

class Net(nn.Module):
    def __init__(self, n_cls=6):
        super(Net, self).__init__()
        self.n_cls = n_cls
        self.num_patches = 15
        extractor = ViTExtractor('dinov2_vits14', 14, device = 'cuda')
        # self.extractor =  torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').cuda()
        self.extractor = torch.hub.load('/home/cpf/.cache/torch/hub/facebookresearch_dinov2_main', 'dinov2_vits14', trust_repo=True, source='local').cuda()

        self.extractor_preprocess = transforms.Normalize(mean=extractor.mean, std=extractor.std)
        self.extractor_layer = 11
        self.extractor_facet = 'token'
        self.sym_rec = EncoderDecoder()
        self.pn2msg = PointNet2MSG(radii_list=[[0.01, 0.02], [0.02,0.04], [0.04,0.08], [0.08,0.16]])

        self.t_mlp = nn.Sequential(
            nn.Conv1d(256, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(128, 3, 1),
        )
        # self.t_weight = nn.Sequential(
        #     nn.Conv1d(256, 256, 1),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(),
        #     nn.Conv1d(256, 128, 1),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Conv1d(128, 1, 1),
        #     nn.Sigmoid()
        # )
        self.t_mlp[-1].bias.data.zero_()
        # self.t_weight[-2].bias.data.zero_()
        self.s_mlp = nn.Sequential(
            nn.Conv1d(256, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(128, 3*self.n_cls, 1),
        )
    def extract_feature(self, rgb_raw):
        
      
        
        
        
        
        rgb_raw = rgb_raw.permute(0,3,1,2)
        
        rgb_raw = self.extractor_preprocess(rgb_raw)
        #import pdb;pdb.set_trace()
        
        with torch.no_grad():
            dino_feature = self.extractor.forward_features(rgb_raw)["x_prenorm"][:,1:]
        
        dino_feature = dino_feature.reshape(dino_feature.shape[0],self.num_patches,self.num_patches,-1)
        
        return dino_feature.contiguous() # b x c x h x w
    def forward(self, inputs):
        rgb = inputs['rgb']
        pts = inputs['pts']
        # temp_pts = inputs['temp_pts']
        cls = inputs['category_label'].long()
        rgb_raw = inputs['rgb_raw']
        b = rgb_raw.shape[0]
        feature = self.extract_feature(rgb_raw).reshape(b,(self.num_patches)**2,-1)
        if 'epoch' in inputs:
            mode = 'train'
            if inputs['epoch'] > 15:
                target_SC = None
            else:
                target_SC = inputs['target_SC'].permute(0,2,1)

        else:
            target_SC = None
            mode = 'test'
        # sacled_pts, norm_centroid, scale = normalize_point_cloud_batch(pts)
        # if target_SC is not None:
        #     target_SC = (target_SC - norm_centroid)/scale[:,None,None]
        #     target_SC = target_SC.permute(0,2,1)
        coarse_pcd, pred_SC, centroid = self.sym_rec(pts.permute(0,2,1), feature, cls, mode, target_SC=target_SC)
        coarse_rgb = torch.cat((rgb, rgb), dim=1)
        coarse_pcd = coarse_pcd.detach()
        centroid = centroid.detach()
        # x = torch.cat([pts, pts, rgb], dim=2)
        x = torch.cat([coarse_pcd, coarse_pcd, coarse_rgb], dim=2)
        x = self.pn2msg(x)

        # weights = self.t_weight(x)

        t = self.t_mlp(x) + coarse_pcd.transpose(1,2)
        t = torch.mean(t, dim=2)
        # weights_normalized = F.softmax(weights, dim=-1)
        # t = torch.bmm(t, weights_normalized.transpose(1,2)).squeeze(-1)
        t = t + centroid.squeeze(1)

        cls = cls.reshape(coarse_pcd.size(0),1,1).expand(coarse_pcd.size(0),1,3).contiguous()
        s = self.s_mlp(x).reshape(coarse_pcd.size(0),self.n_cls,3,coarse_pcd.size(1)).mean(3)
        # s = self.s_mlp(x)
        # s = torch.bmm(s, weights_normalized.transpose(1,2)).reshape(coarse_pcd.size(0),self.n_cls,3) # bs x nc x 3 x cate_npts
        s = torch.gather(s,1,cls).squeeze(1)

        end_points = {}
        end_points['translation'] = t
        end_points['size'] = s
        end_points['pred_SC'] = pred_SC

        return end_points

class Loss(nn.Module):
    def __init__(self, cfg):
        super(Loss, self).__init__()
        self.cfg = cfg
        self.criterion = nn.L1Loss()

    def forward(self, pred, gt):
        loss_t = self.criterion(pred['translation'], gt['translation_label'])
        loss_s = self.criterion(pred['size'], gt['size_label'])
        # target_SC = (gt['target_SC'] - pred['norm_centroid'])/pred['scale'][:,None,None]
        loss_SC = self.criterion(pred['pred_SC'], gt['target_SC'])
        if loss_SC > 5:
            # raise ValueError("loss_SC is too large, please check the input data")
            loss_SC = 0.0
        loss =  self.cfg.t_weight*loss_t+self.cfg.s_weight*loss_s+loss_SC
        return {
            'loss': loss,
            't': loss_t,
            's': loss_s,
            'SC': loss_SC,
        }


class EncoderDecoder(nn.Module):
    def __init__(self, num_cate=6):
        super(EncoderDecoder, self).__init__()
        self.num_cate = num_cate
        self.posefeat = PointEncoder()

        # self.category_local = nn.Sequential(
        #     nn.Conv1d(3, 64, 1),
        #     nn.ReLU(),
        #     nn.Conv1d(64, 64, 1),
        #     nn.ReLU(),
        #     nn.Conv1d(64, 64, 1),
        #     nn.ReLU(),
        # )
        # self.category_global = nn.Sequential(
        #     nn.Conv1d(64, 128, 1),
        #     nn.ReLU(),
        #     nn.Conv1d(128, 1024, 1),
        #     nn.ReLU(),
        #     nn.AdaptiveAvgPool1d(1),
        # )
        self.rgb_avg = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(384, 1024, 1),
        )



        self.sym = nn.Sequential(
            nn.Conv1d((3 + 1024 + 896), 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3*self.num_cate, 1),
        )

    def forward(self, obsv_pcd, rgb_f, cate_id, mode, target_SC = None):
        '''
        :param x: (B,3,N)
        :param prior: (B,3,N_k)
        :param cate_id: (B)
        :param mode: 'train' or 'test'
        :param sym_part_gt: (B,3,N)
        :return:
        '''

        # temp_pcd = deepcopy(prior)
        # obsv_pcd = deepcopy(x)


        # bs, dim, temp_npts = temp_pcd.shape
        bs, _, obsv_npts = obsv_pcd.shape

        inst_global = self.posefeat(obsv_pcd) # (B,896,1)

        # temp_local = self.category_local(temp_pcd)  # (B,64,N)
        # temp_global = self.category_global(temp_local)  # (B,1024,1)
        seg_global = self.rgb_avg(rgb_f.transpose(-1,1))

        sym_feat = torch.cat((obsv_pcd,
                           seg_global.repeat(1, 1, obsv_npts),
                           inst_global.repeat(1, 1, obsv_npts)),
                          dim=1)  # bs x (3+1024+896)) x 2048

        index = cate_id.squeeze() + torch.arange(bs, dtype=torch.long).cuda() * self.num_cate

        # Symmetric Correspondence
        pred_SC = self.sym(sym_feat)
        pred_SC = pred_SC.view(-1, 3, obsv_npts).contiguous() # bs, nc*3, cate_npts -> bs*nc, 3, cate_npts
        pred_SC = torch.index_select(pred_SC, 0, index).contiguous()  # bs x 3 x inst_npts

        if mode == 'train':
            if target_SC is not None:
                coarse_pcd = torch.cat((obsv_pcd, target_SC), dim=2)
            else:
                coarse_pcd = torch.cat((obsv_pcd, pred_SC), dim=2)
        else:
            coarse_pcd = torch.cat((obsv_pcd, pred_SC), dim=2)
        # coarse_pcd = coarse_pcd * scale[:,None,None]+norm_centroid.permute(0,2,1)
        coarse_pcd, centroid = pc_centralize(coarse_pcd)


        return coarse_pcd.permute(0,2,1), pred_SC.permute(0,2,1), centroid.permute(0,2,1)

class PointEncoder(nn.Module):
    def __init__(self):
        super(PointEncoder, self).__init__()
        self.e_conv1 = nn.Conv1d(3, 64, 1)  # (B,3,N)
        self.e_conv2 = nn.Conv1d(64, 64, 1)
        self.e_conv3 = nn.Conv1d(64, 128, 1)
        self.e_conv4 = nn.Conv1d(128, 256, 1)
        self.e_conv5 = nn.Conv1d(256, 512, 1)

        self.bn1 = nn.InstanceNorm1d(64)
        self.bn2 = nn.InstanceNorm1d(64)
        self.bn3 = nn.InstanceNorm1d(128)
        self.bn4 = nn.InstanceNorm1d(256)
        self.bn5 = nn.InstanceNorm1d(512)

    def forward(self, x):
        '''
        :param x: (B,3,N)
        :return:
        '''
        x = F.relu(self.bn1(self.e_conv1(x)))
        x = F.relu(self.bn2(self.e_conv2(x)))
        x = F.relu(self.bn3(self.e_conv3(x)))  # (B,128,N)
        maxpool_128, _ = torch.max(x, 2)  # (B,128)
        x = F.relu(self.bn4(self.e_conv4(x)))
        maxpool_256, _ = torch.max(x, 2)
        x = F.relu(self.bn5(self.e_conv5(x)))
        maxpool_512, _ = torch.max(x, 2)

        Feature = [maxpool_128, maxpool_256, maxpool_512]
        inst_global = torch.cat(Feature, 1).unsqueeze(2)  # (B,896,1)

        return inst_global

def pc_centralize(pcd):
    '''
    :param pcd: (B,3,N)
    :return: (B,3,N)
    '''
    pc = pcd.clone()
    centroid = torch.mean(pc, dim=2).unsqueeze(2) #(B,3,1)
    pc = pc - centroid

    return pc, centroid

def normalize_point_cloud_batch(pc):
    """
    将批次中的点云归一化到 (-1, 1) 范围内，并返回中心和长轴长度。

    参数:
        pc (torch.Tensor): 输入的点云，形状为 (batch_size, N, 3)。

    返回:
        tuple: 归一化后的点云 (torch.Tensor)，点云的中心 (torch.Tensor)，长轴的长度 (torch.Tensor)。
    """
    # 计算每个点云的中心
    centroid = pc.mean(dim=1, keepdim=True)  # 形状为 (batch_size, 1, 3)
    
    # 将点云中心置于原点
    pc_centered = pc - centroid  # 归一化后的点云

    # 计算每个点云的长轴长度
    m = torch.max(torch.sqrt(torch.sum(pc_centered ** 2, dim=2)), dim=1)[0]  # 形状为 (batch_size,)
    m = torch.clamp(m, min=0.05)
    # 归一化点云到 (-1, 1)
    pc_normalized = pc_centered / m.view(-1, 1, 1)  # 重新调整 m 的形状以便于广播

    return pc_normalized, centroid, m# import torch
