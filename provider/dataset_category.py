import os,json
import math
import re
import cv2
import glob
import numpy as np
import _pickle as cPickle
from PIL import Image
from scipy.spatial.transform import Rotation as R
import time

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import open3d as o3d
import point_cloud_utils as pcu

from utils.data_utils import (
    load_depth,
    load_composed_depth,
    get_bbox,
    fill_missing,
    get_bbox_from_mask,
    rgb_add_noise,
    random_rotate,
    random_scale,
)

defaultTrainconfig = {
   
  'data_dir': './data',
  'sample_num': 2048,
  'random_rotate': True,
  'angle_range': 20
}

class TrainingDataset(Dataset):
    def __init__(self,
            config, 
            dataset='REAL275',
            mode='ts',
            num_img_per_epoch=-1,
            resolution=64,
            ds_rate=2,
            for_sim_feature = False,
            num_patches = 15,
            category = 0
    ):
        
        np.random.seed(0)
        self.category = category
        assert mode in ['ts','r','sim']
        self.config = config
        self.dataset = dataset
        self.mode = mode
        self.num_img_per_epoch = num_img_per_epoch

        self.resolution = resolution
        self.ds_rate = ds_rate
        self.for_sim_feature = for_sim_feature
        self.num_patches = num_patches
        try: 
            self.sample_num = self.config.sample_num
            self.data_dir = config.data_dir
        except:
            self.sample_num = self.config['sample_num']
            self.data_dir = config['data_dir']
        

        self.invalid_index = []
        syn_img_path = 'camera/train_list.txt'
        self.syn_intrinsics = [577.5, 577.5, 319.5, 239.5]
        self.syn_img_list = [os.path.join(syn_img_path.split('/')[0], line.rstrip('\n'))
                        for line in open(os.path.join(self.data_dir, syn_img_path))]
        #self.syn_img_list = []#CHANGE THIS
        #import pdb;pdb.set_trace()
        print('{} synthetic images are found.'.format(len(self.syn_img_list)))

        syn_category_path = 'camera/train_category_dict.json'
        self.syn_category_dict = json.load(open(os.path.join(self.data_dir, syn_category_path)))
        syn_category_dict_tmp = self.syn_category_dict
        for cls in syn_category_dict_tmp.keys():
                syn_category_dict_tmp[cls] = [[x[0], x[1], 'syn'] for x in syn_category_dict_tmp[cls]]
        self.reference_category_dict = syn_category_dict_tmp

        if self.dataset == 'REAL275':
            real_img_path = 'real/train_list.txt'
            self.real_intrinsics = [591.0125, 590.16775, 322.525, 244.11084]
            self.real_img_list = [os.path.join(real_img_path.split('/')[0], line.rstrip('\n'))
                            for line in open(os.path.join(self.data_dir, real_img_path))]
            print('{} real images are found.'.format(len(self.real_img_list)))
            real_category_path = 'real/train_category_dict.json'
            self.real_category_dict = json.load(open(os.path.join(self.data_dir, real_category_path)))
            real_category_dict_tmp = self.real_category_dict
            for cls in real_category_dict_tmp.keys():
                real_category_dict_tmp[cls] = [[x[0], x[1], 'real'] for x in real_category_dict_tmp[cls]]
            self.reference_category_dict = {cat:self.reference_category_dict[cat] + real_category_dict_tmp[cat] for cat in self.reference_category_dict.keys()}
        
        self.cls_list = sorted(list(self.reference_category_dict.keys()))
        self.xmap = np.array([[i for i in range(640)] for j in range(480)])
        self.ymap = np.array([[j for i in range(640)] for j in range(480)])
        self.sym_ids = [0, 1, 3]    # 0-indexed
        self.norm_scale = 1000.0    # normalization scale
        self.colorjitter = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        # self.transform = transforms.Compose([transforms.ToTensor(),
        #                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                                           std=[0.229, 0.224, 0.225])])#xinjia
        self.id2cat_name = {'1': 'bottle', '2': 'bowl', '3': 'camera', '4': 'can', '5': 'laptop', '6': 'mug'}
        self.temp_pcd_list = []
        for i in range(6):
            temp_pcd_path = os.path.join('data/template_FPS', f'{self.id2cat_name[str(i + 1)]}_fps_36_normalized.obj')
            temp_pcd, _ = load_obj(temp_pcd_path)
            self.temp_pcd_list.append(temp_pcd)
        # with open(os.path.join(self.data_dir, 'real/train/mug_handle.pkl'), 'rb') as f:
        #     self.mug_sym = cPickle.load(f)
        
        self.feature_instance_list = []
        for cls in self.cls_list:
            num_instance = len(self.reference_category_dict[cls])
            
            self.feature_instance_list+=[(cls, i) for i in range(num_instance)]

            
            
            

        if self.num_img_per_epoch != -1:
            self.reset()

    def __len__(self):
        if self.for_sim_feature:
            return len(self.feature_instance_list)

        if self.mode == 'ts':
            if self.num_img_per_epoch == -1:
                if self.dataset == 'REAL275':
                    return len(self.syn_img_list) + len(self.real_img_list)
                else:
                    return len(self.syn_img_list)
            else:
                return self.num_img_per_epoch
        elif self.mode in ['r','sim']:
            if self.num_img_per_epoch == -1:
                num_syn_instance = sum([len(self.syn_category_dict[k]) for k in self.syn_category_dict.keys()])
                    
                if self.dataset == 'REAL275':
                    num_real_instance =  sum([len(self.syn_category_dict[k]) for k in self.syn_category_dict.keys()])
                    return num_syn_instance + num_real_instance
                else:
                    return num_syn_instance
            else:
                return len(self.instance_index)
        
    def reset(self):
        assert self.num_img_per_epoch != -1
        def choice(x, y):
            if x<=y:
                return np.random.choice(x, y)
            else:
                return np.random.choice(x, y, replace = False)
        if self.mode == 'ts':
            if self.dataset == 'REAL275':
                num_syn_img = len(self.syn_img_list)
                num_syn_img, num_real_img = len(self.syn_img_list), len(self.real_img_list)
                num_syn_img_per_epoch = int(self.num_img_per_epoch*0.75)
                #num_syn_img_per_epoch = 0 #CHANGE THIS
                num_real_img_per_epoch = self.num_img_per_epoch - num_syn_img_per_epoch
                syn_img_index = choice(num_syn_img, num_syn_img_per_epoch)
                real_img_index = choice(num_real_img, num_real_img_per_epoch)
                real_img_index = -real_img_index - 1
                self.img_index = np.hstack([syn_img_index, real_img_index])

            else:
                num_syn_img = len(self.syn_img_list)
                num_syn_img_per_epoch = int(self.num_img_per_epoch)
                syn_img_index = choice(num_syn_img, num_syn_img_per_epoch)
                self.img_index = syn_img_index
            #import pdb;pdb.set_trace()
            
            np.random.shuffle(self.img_index)
        elif self.mode in ['r', 'sim']:            

            if self.dataset == 'REAL275':
                num_instance_per_epoch = self.num_img_per_epoch
                syn_category_num_dict = {cat: len(self.syn_category_dict[cat]) 
                                            for cat in self.syn_category_dict.keys()}
                num_syn_instance = sum([len(self.syn_category_dict[k]) for k in self.syn_category_dict.keys()])
                num_syn_instance_per_epoch = int(num_instance_per_epoch*0.75)
                syn_category_ratio = {cat:len(self.syn_category_dict[cat])/ num_syn_instance
                                        for cat in self.syn_category_dict.keys()}
                syn_category_sample_num = {cat: int(num_syn_instance_per_epoch*syn_category_ratio[cat]) 
                                            for cat in syn_category_ratio.keys()}
                syn_instance_index_dict = {cat: choice(syn_category_num_dict[cat], syn_category_sample_num[cat]) 
                                    for cat in self.syn_category_dict.keys()}
                # syn_instance_index = np.concatenate(
                #         [np.concatenate(
                #         (syn_instance_index_dict[cat], np.array([[int(cat)]*syn_instance_index_dict[cat].shape[0]]).T) ,axis = 1)
                #                         for cat in syn_instance_index_dict.keys()],axis = 0
                #     )
                # self.instance_index = syn_instance_index
                
                real_category_num_dict = {cat: len(self.real_category_dict[cat]) 
                                         for cat in self.real_category_dict.keys()}
                
                num_real_instance =  sum([len(self.real_category_dict[k]) for k in self.real_category_dict.keys()])
                
                #num_syn_img_per_epoch = 0 #CHANGE THIS
                num_real_instance_per_epoch = num_instance_per_epoch - num_syn_instance_per_epoch
                real_category_ratio = {cat:len(self.real_category_dict[cat])/ num_real_instance
                                       for cat in self.real_category_dict.keys()}
                real_category_sample_num = {cat: int(num_real_instance_per_epoch* real_category_ratio[cat]) 
                                           for cat in real_category_ratio.keys()}
                
                real_instance_index_dict = {cat: - choice(real_category_num_dict[cat], real_category_sample_num[cat])-1
                                 for cat in self.real_category_dict.keys()}

                instance_index_dict = {cat: np.concatenate([real_instance_index_dict[cat], syn_instance_index_dict[cat]]).reshape(-1,1) for cat in real_instance_index_dict.keys()}
                for cat in instance_index_dict.keys():
                    np.random.shuffle(instance_index_dict[cat])
                
                # real_instance_index = np.concatenate(
                #     [np.concatenate(
                #     (-real_instance_index_dict[cat] - 1, np.array([[int(cat)]*real_instance_index_dict[cat].shape[0]]).T) ,axis = 1)
                #                       for cat in real_instance_index_dict.keys()],axis = 0
                # )
                
                # self.instance_index = np.vstack([syn_instance_index, real_instance_index])
                self.instance_index = np.concatenate(
                    [np.concatenate(
                    (instance_index_dict[cat] , np.array([[int(cat)]*instance_index_dict[cat].shape[0]]).T) ,axis = 1)
                                      for cat in instance_index_dict.keys()],axis = 0
                )
        
            else:
                num_instance_per_epoch = self.num_img_per_epoch
                syn_category_num_dict = {cat: len(self.syn_category_dict[cat]) 
                                            for cat in self.syn_category_dict.keys()}
                num_syn_instance = sum([len(self.syn_category_dict[k]) for k in self.syn_category_dict.keys()])
                num_syn_instance_per_epoch = int(num_instance_per_epoch)
                syn_category_ratio = {cat:len(self.syn_category_dict[cat])/ num_syn_instance
                                        for cat in self.syn_category_dict.keys()}
                syn_category_sample_num = {cat: int(num_syn_instance_per_epoch*syn_category_ratio[cat]) 
                                            for cat in syn_category_ratio.keys()}
                syn_instance_index_dict = {cat: choice(syn_category_num_dict[cat], syn_category_sample_num[cat]).reshape(-1,1)
                                    for cat in self.syn_category_dict.keys()}
                syn_instance_index = np.concatenate(
                        [np.concatenate(
                        (syn_instance_index_dict[cat], np.array([[int(cat)]*syn_instance_index_dict[cat].shape[0]]).T) ,axis = 1)
                                        for cat in syn_instance_index_dict.keys()],axis = 0
                    )
                self.instance_index = syn_instance_index
            
             
            
            np.random.shuffle(self.instance_index)
        else:
            assert False
     

    def __getitem__(self, index):
        if self.for_sim_feature:
            while True:
                cls, idx = self.feature_instance_list[index]

                data_dict =  self._read_instance_from_category_dict(cls, idx)
                if data_dict is None:
                    index +=1
                    self.invalid_index.append(index)
                    continue
                data_dict['index'] = torch.IntTensor([idx]).long()
                data_dict['cls'] = torch.IntTensor([int(cls)]).long()
                return data_dict
                
        
        if self.mode =='ts':
            while True:
                image_index = self.img_index[index]
                data_dict = self._read_instance(image_index)
                #print('READ DATA',index,"/",self.__len__(),time.time()-st)
                if data_dict is None:
                    index = np.random.randint(self.__len__())
                    continue
                return data_dict
        elif self.mode in ['r','sim']:
            while True:
                index, cat = self.instance_index[index]
                data_dict = self._read_instance_cat(index, cat)
                #print('READ DATA',index,"/",self.__len__(),time.time()-st)
                if data_dict is None:
                    index = np.random.randint(self.__len__())
                    continue
                return data_dict

    def _read_instance_cat(self, index,cat):
        assert self.mode in [ 'r','sim']
        def get_data(index):
            if index>=0:
                instance_type = 'syn'
                img_path, instance_id, _ = self.syn_category_dict[str(cat)][index]
                cam_fx, cam_fy, cam_cx, cam_cy = self.syn_intrinsics
            else:
                instance_type = 'real'
                index = -index-1
                
                img_path, instance_id ,_= self.real_category_dict[str(cat)][index]
                cam_fx, cam_fy, cam_cx, cam_cy = self.real_intrinsics
            return self._load_data(img_path,
                                        instance_type, 
                                        cam_cx, cam_cy,cam_fx, cam_fy, instance_id)
        tuple = get_data(index)   
        if tuple is None :
            return None
        pts, rgb, translation, \
        rotation, size, cat_id, asym_flag, \
        rmin, rmax, cmin, cmax, choose, \
            rgb_raw, pts_raw, mask, rand_rotation, pts_yuanshi2= tuple


        v = rotation[:,2] / (np.linalg.norm(rotation[:,2])+1e-8)
        rho = np.arctan2(v[1], v[0])#alpha
        if v[1]<0:
            rho += 2*np.pi
        phi = np.arccos(v[2])#

        vp_rotation = np.array([
            [np.cos(rho),-np.sin(rho),0],
            [np.sin(rho), np.cos(rho),0],
            [0,0,1]
        ]) @ np.array([
            [np.cos(phi),0,np.sin(phi)],
            [0,1,0],
            [-np.sin(phi),0,np.cos(phi)],
        ])
        ip_rotation = vp_rotation.T @ rotation

        rho_label = int(rho / (2*np.pi) * (self.resolution//self.ds_rate))
        phi_label = int(phi/np.pi*(self.resolution//self.ds_rate)) 
        


        ret_dict = {}
        ret_dict['rgb'] = torch.FloatTensor(rgb)
        ret_dict['rgb_raw'] = torch.FloatTensor(rgb_raw)
        ret_dict['pts'] = torch.FloatTensor(pts)
        ret_dict['pts_raw'] = torch.FloatTensor(pts_raw)
        # ret_dict['rmin'] = torch.IntTensor([rmin_first, rmin_second]).long()
        # ret_dict['rmax'] = torch.IntTensor([rmax_first, rmax_second]).long()
        # ret_dict['cmin'] = torch.IntTensor([cmin_first, cmin_second]).long()
        # ret_dict['cmax'] = torch.IntTensor([cmax_first, cmax_second]).long()
        ret_dict['choose'] = torch.IntTensor(choose).long()
        ret_dict['mask'] = torch.IntTensor(mask).long()
        
        ret_dict['category_label'] = torch.IntTensor([cat_id]).long()
        ret_dict['asym_flag'] = torch.FloatTensor([asym_flag])
        ret_dict['translation_label'] = torch.FloatTensor(translation)
        ret_dict['rotation_label'] = torch.FloatTensor(rotation)
        
        ret_dict['size_label'] = torch.FloatTensor(size)

        ret_dict['rho_label'] = torch.IntTensor([rho_label]).long()
        ret_dict['phi_label'] = torch.IntTensor([phi_label]).long()
        ret_dict['vp_rotation_label'] = torch.FloatTensor(vp_rotation)
        ret_dict['ip_rotation_label'] = torch.FloatTensor(ip_rotation)
        ret_dict['rand_rotation'] = torch.FloatTensor(rand_rotation)
        ret_dict['ptsg'] = torch.FloatTensor(pts_yuanshi2[:300,:])
        normals = self.cal_normal(ret_dict['pts'][:300,:], ret_dict['rand_rotation'], ret_dict['translation_label'], ret_dict['size_label'])
        ret_dict['normal'] = normals
        

        return ret_dict
        
    def get_sym_info(self, c, mug_handle=1):
        #  sym_info  c0 : face classfication  c1, c2, c3:Three view symmetry, correspond to xy, xz, yz respectively
        # c0: 0 no symmetry 1 axis symmetry 2 two reflection planes 3 unimplemented type
        #  Y axis points upwards, x axis pass through the handle, z axis otherwise
        #
        # for specific defination, see sketch_loss
        if c == 'bottle':
            sym = np.array([1, 1, 0, 1], dtype=np.int)
        elif c == 'bowl':
            sym = np.array([1, 1, 0, 1], dtype=np.int)
        elif c == 'camera':
            # sym = np.array([0, 0, 0, 0], dtype=np.int)
            sym = np.array([0, 1, 0, 0], dtype=np.int)
        elif c == 'can':
            sym = np.array([1, 1, 1, 1], dtype=np.int)
        elif c == 'laptop':
            sym = np.array([0, 1, 0, 0], dtype=np.int)
        elif c == 'mug' and mug_handle == 1:
            sym = np.array([0, 1, 0, 0], dtype=np.int)  # for mug, we currently mark it as no symmetry
        elif c == 'mug' and mug_handle == 0:
            sym = np.array([1, 0, 0, 0], dtype=np.int)
        else:
            sym = np.array([0, 0, 0, 0], dtype=np.int)
        return sym
    
    def _load_data(self,img_path,img_type, cam_cx, cam_cy,cam_fx, cam_fy, instance_id = -1, without_noise = False):
        #import pdb;pdb.set_trace()
        if self.mode == 'SIM':
            without_noise = True
        if self.dataset == 'REAL275':
            depth = load_composed_depth(img_path)
            depth = fill_missing(depth, self.norm_scale, 1)

        else:
            depth = load_depth(img_path)

        # mask
        with open(img_path + '_label.pkl', 'rb') as f:
            gts = cPickle.load(f)
        #print("READ MASK:",img_path,time.time()-st)
        
        assert(len(gts['class_ids'])==len(gts['instance_ids']))
        mask = cv2.imread(img_path + '_mask.png')[:, :, 2] #480*640

        
        if instance_id == -1:
            num_instance = len(gts['instance_ids'])
            instance_id = np.random.randint(0, num_instance)
        cat_id = gts['class_ids'][instance_id] - 1 # convert to 0-indexed
        rmin, rmax, cmin, cmax = get_bbox(gts['bboxes'][instance_id])
        mask = np.equal(mask, gts['instance_ids'][instance_id])
        mask = np.logical_and(mask , depth > 0)
        mask = mask[rmin:rmax, cmin:cmax]
        h,w = mask.shape
        # choose
        choose = mask.flatten().nonzero()[0]
        if len(choose)<=5:
            return None
        elif len(choose) <= self.sample_num:
            choose_idx = np.random.choice(np.arange(len(choose)), self.sample_num)
        else:
            choose_idx = np.random.choice(np.arange(len(choose)), self.sample_num, replace=False)
        choose = choose[choose_idx]

        # pts
        pts2 = depth.copy()[rmin:rmax, cmin:cmax].reshape((-1)) / self.norm_scale
        pts0 = (self.xmap[rmin:rmax, cmin:cmax].reshape((-1)) - cam_cx) * pts2 / cam_fx
        pts1 = (self.ymap[rmin:rmax, cmin:cmax].reshape((-1))- cam_cy) * pts2 / cam_fy
        pts = np.transpose(np.stack([pts0, pts1, pts2]), (1,0)).astype(np.float32) # 480*640*3
        if not without_noise:
            pts = pts + np.clip(0.001*np.random.randn(pts.shape[0], 3), -0.005, 0.005)

        pts_raw = pts#.reshape(h,w,3)
        
        

        

        # rgb
        rgb = cv2.imread(img_path + '_color.png')[:, :, :3]
        rgb = rgb[:, :, ::-1] #480*640*3
        rgb_raw = rgb[rmin:rmax, cmin:cmax]
        
        
        if not without_noise:
            rgb_raw = self.colorjitter(Image.fromarray(np.uint8(rgb_raw)))
            
        rgb_raw = np.array(rgb_raw)
        if img_type == 'syn' and not without_noise:
            rgb_raw = rgb_add_noise(rgb_raw)
        # rgb_raw = np.array(self.transform(np.array(rgb_raw)))
        rgb_raw = rgb_raw.astype(np.float32).reshape((-1,3))/ 255.0
        rgb_raw = rgb_raw.astype(np.float32).reshape((-1,3))
        rgb = rgb_raw[choose] 
        

        # gt
        translation = gts['translations'][instance_id].astype(np.float32)
        rotation = gts['rotations'][instance_id].astype(np.float32)
        size = gts['scales'][instance_id] * gts['sizes'][instance_id].astype(np.float32)


        if hasattr(self.config, 'random_rotate') and self.config.random_rotate and not without_noise:
            pts_raw, rotation, rand_rotation = random_rotate(pts_raw, rotation, translation, self.config.angle_range, return_rand_rotation = True)
        else:
            rand_rotation = np.eye(3)
        if self.mode == 'ts':
            pts = pts_raw[choose][:1024]
            rgb = rgb[:1024]
            # temp_pcd_path = os.path.join('data/template_FPS', f'{self.id2cat_name[str(cat_id + 1)]}_fps_36_normalized.obj')
            # temp_pcd, _ = load_obj(temp_pcd_path)
            temp_pcd = self.temp_pcd_list[cat_id]
            # if (cat_id+1) == 6 and img_type == 'real':
            #     handle_tmp_path = img_path.split('/')
            #     scene_label = handle_tmp_path[-2] + '_res'
            #     img_id = int(handle_tmp_path[-1])
            #     mug_handle = self.mug_sym[scene_label][img_id]
            # else:
            mug_handle = 1
            sym_info = self.get_sym_info(self.id2cat_name[str(cat_id + 1)], mug_handle=mug_handle)
            pts, size = random_scale(pts, size, rotation, translation)

            center = np.mean(pts, axis=0)
            pts = pts - center[np.newaxis, :]
            translation = translation - center

            noise_t = np.random.uniform(-0.02, 0.02, 3)
            pts = pts + noise_t[None, :]
            translation = translation + noise_t

            points_re_cano = (pts - translation[None, :]) @ rotation
            if sym_info[0] == 1 and sym_info[1:].sum() > 0: #For y axis reflection, can, bowl, bottle
                gt_re_points = points_re_cano * np.array([-1, 1, -1], dtype=points_re_cano.dtype).reshape(-1, 3)
                gt_PC = gt_re_points @ rotation.T + translation[None, :]
            elif sym_info[0] == 0 and sym_info[1] == 1: #For yx axis reflection, laptop, mug, camera
                gt_re_points = points_re_cano * np.array([1, 1, -1], dtype=points_re_cano.dtype).reshape(-1, 3)
                gt_PC = gt_re_points @ rotation.T + translation[None, :]
            # elif sym_info[0] == 1 and sym_info[1:].sum() == 0:#For no symmetry, mug handle=0
            #     gt_PC = np.zeros_like(points_re_cano)
            rgb_raw = rgb_raw.reshape(h,w,3)
            
            
            rgb_raw = cv2.resize(rgb_raw, dsize=(self.num_patches*14,self.num_patches*14), interpolation=cv2.INTER_NEAREST)
            return pts, rgb, translation, rotation, size, cat_id, sym_info, gt_PC, temp_pcd, rgb_raw.copy()
        elif self.mode in  ['r', 'sim']:
            
            noise_t = np.random.uniform(-0.02, 0.02, 3)
            noise_s = np.random.uniform(0.8, 1.2, 1)
            if without_noise:
                pts_yuanshi = pts_raw.copy()
                pts_raw = pts_raw - translation[None, :]
                pts_raw = pts_raw / np.linalg.norm(size)
                
            else:
                pts_yuanshi = (pts_raw - noise_t[None, :])* noise_s
                pts_raw = pts_raw - translation[None, :] - noise_t[None, :]
                pts_raw = pts_raw / np.linalg.norm(size) * noise_s
                

            if cat_id in self.sym_ids:
                theta_x = rotation[0, 0] + rotation[2, 2]
                theta_y = rotation[0, 2] - rotation[2, 0]
                r_norm = math.sqrt(theta_x**2 + theta_y**2)
                s_map = np.array([[theta_x/r_norm, 0.0, -theta_y/r_norm],
                                    [0.0,            1.0,  0.0           ],
                                    [theta_y/r_norm, 0.0,  theta_x/r_norm]])
                rotation = rotation @ s_map

                asym_flag = 0.0
            else:
                asym_flag = 1.0

            # transform ZXY system to XYZ system
            pts = pts_raw[choose]
            pts_yuanshi2 = pts_yuanshi[choose]
            pts_raw = pts_raw.reshape(h,w,3)
            rgb_raw = rgb_raw.reshape(h,w,3)
            
            rotation = rotation[:, (2,0,1)]#相当于原来的Z轴变X轴，X轴变Y轴，Y轴变Z轴
            
            rgb_raw = cv2.resize(rgb_raw, dsize=(self.num_patches*14,self.num_patches*14), interpolation=cv2.INTER_NEAREST)
            pts_raw = np.where((mask == 0)[:,:,None],np.nan, pts_raw)
            pts_raw = cv2.resize(pts_raw, dsize=(self.num_patches,self.num_patches), interpolation=cv2.INTER_NEAREST)
            mask = np.logical_not(np.isnan(pts_raw)).all(axis = -1)
            # choose
            choose = mask.flatten().nonzero()[0]
            if len(choose)<=5:
                return None
            elif len(choose) <= self.sample_num:
                choose_idx = np.random.choice(np.arange(len(choose)), self.sample_num)
            else:
                choose_idx = np.random.choice(np.arange(len(choose)), self.sample_num, replace=False)
            choose = choose[choose_idx]

            
            


            return pts, rgb, translation, rotation, size, cat_id, asym_flag, \
                rmin, rmax, cmin, cmax, choose, rgb_raw.copy(), pts_raw.copy(), mask.copy() , rand_rotation, pts_yuanshi2
        else: 
            assert False


    def _read_instance(self, image_index):
        assert self.mode == 'ts'
        if image_index>=0:
            img_type = 'syn'
            img_path = os.path.join(self.data_dir, self.syn_img_list[image_index])
            cam_fx, cam_fy, cam_cx, cam_cy = self.syn_intrinsics
        else:
            img_type = 'real'
            image_index = -image_index-1
            img_path = os.path.join(self.data_dir, self.real_img_list[image_index])
            cam_fx, cam_fy, cam_cx, cam_cy = self.real_intrinsics
        tuple_instance = self._load_data(img_path,
                                                        img_type, 
                                                        cam_cx, cam_cy,cam_fx, cam_fy)
        if tuple_instance is None:
            return None
        pts, rgb, translation, rotation, size, cat_id, sym_info, gt_PC_sym, temp_pcd, rgb_raw = tuple_instance
        
        
        ret_dict = {}
        ret_dict['pts'] = torch.FloatTensor(pts)
        ret_dict['rgb'] = torch.FloatTensor(rgb)
        ret_dict['category_label'] = torch.IntTensor([cat_id]).long()
        ret_dict['translation_label'] = torch.FloatTensor(translation)
        ret_dict['size_label'] = torch.FloatTensor(size)
        ret_dict['sym_info'] = torch.FloatTensor(sym_info)
        ret_dict['target_SC'] = torch.FloatTensor(gt_PC_sym)
        ret_dict['rgb_raw'] = torch.FloatTensor(rgb_raw)

        return ret_dict
    def _read_instance_from_category_dict(self, cls, idx ):
        img_path, instance_id, img_type = self.reference_category_dict[str(cls)][idx]
        
        if img_type == 'real':
            
            
            cam_fx, cam_fy, cam_cx, cam_cy = self.real_intrinsics

        else:
            
            
            cam_fx, cam_fy, cam_cx, cam_cy = self.syn_intrinsics
        tuple_instance = self._load_data(img_path,
                                        img_type, 
                                        cam_cx, cam_cy,cam_fx, cam_fy, instance_id = instance_id,without_noise=True)
        if tuple_instance is None:
            return None

        
        pts, rgb, translation, rotation, size, cat_id, asym_flag, \
                rmin, rmax, cmin, cmax, choose, rgb_raw, pts_raw, mask, rand_rotation = tuple_instance
        
        ret_dict = {}
        ret_dict['pts'] = torch.FloatTensor(pts)
        ret_dict['rgb'] = torch.FloatTensor(rgb)
        ret_dict['rgb_raw'] = torch.FloatTensor(rgb_raw)
        ret_dict['pts_raw'] = torch.FloatTensor(pts_raw)
        ret_dict['category_label'] = torch.IntTensor([cat_id]).long()
        ret_dict['translation_label'] = torch.FloatTensor(translation)
        ret_dict['size_label'] = torch.FloatTensor(size)
        ret_dict['rotation_label'] = torch.FloatTensor(rotation)
        ret_dict['rmin'] = torch.IntTensor([rmin]).long()
        ret_dict['rmax'] = torch.IntTensor([rmax]).long()
        ret_dict['cmin'] = torch.IntTensor([cmin]).long()
        ret_dict['cmax'] = torch.IntTensor([cmax]).long()
        ret_dict['choose'] = torch.IntTensor(choose).long()
        ret_dict['mask'] = torch.IntTensor(mask).long()
        return ret_dict
    def get_ref_data(self, clss, indexes):
        data_list = []
        assert len(clss) == len(indexes)
        for cls, index in zip(clss, indexes):
            assert len(index) == 1
            
            index = int(index[0].item())
            cls = int(cls.item())
            data_list.append( self._read_instance_from_category_dict(cls, index) )
        ret_dict = {}
        for k in data_list[0].keys():
            ret_dict[k] = torch.stack([d[k] for d in data_list], dim = 0)
        return ret_dict

