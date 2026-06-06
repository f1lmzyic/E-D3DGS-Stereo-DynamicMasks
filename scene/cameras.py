#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getProjectionMatrixCV
from utils.graphics_utils import fov2focal, pix2ndc
from kornia import create_meshgrid
import random 
from torchvision import transforms
from PIL import Image


class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", near=0.01, far=100.0, timestamp=0.0, rayo=None, rayd=None, rays=None, cxr=0.0,cyr=0.0,
                 cam_no=None, frame_no=None, image_path=None, img_wh=None, dynamic_mask=None, dynamic_mask_path=None,
                 motion_prior=None, motion_prior_path=None, depth_map=None, depth_path=None):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.time = timestamp
        self.cam_no = cam_no
        self.frame_no = frame_no

        self.transform = transforms.ToTensor()
        self.gt_alpha_mask = gt_alpha_mask
        self.img_wh = img_wh
        self.image_path = image_path
        self.dynamic_mask = dynamic_mask
        self.dynamic_mask_path = dynamic_mask_path
        self.motion_prior = motion_prior
        self.motion_prior_path = motion_prior_path
        self.depth_map = depth_map
        self.depth_path = depth_path
        
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        # image is real image 
        if not isinstance(image, tuple) and image is not None:
            if "camera_" not in image_name:
                self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
            else:
                self.original_image = image.clamp(0.0, 1.0).half().to(self.data_device)
            self.image_width = self.original_image.shape[2]
            self.image_height = self.original_image.shape[1]
            if gt_alpha_mask is not None:
                self.original_image *= gt_alpha_mask.to(self.data_device)
            else:
                self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        elif isinstance(image, tuple) and image is not None:
            self.image_width = image[0]
            self.image_height = image[1]
            self.original_image = None
        
        else: # image: None
            self.image_width = None
            self.image_height = None
            self.original_image = None
        
        if self.dynamic_mask is None and self.dynamic_mask_path is not None and self.image_width is not None and self.image_height is not None:
            self.load_dynamic_mask()
        if self.motion_prior is None and self.motion_prior_path is not None and self.image_width is not None and self.image_height is not None:
            self.load_motion_prior()
        if self.depth_map is None and self.depth_path is not None and self.image_width is not None and self.image_height is not None:
            self.load_depth_map()


        self.zfar = 100.0
        self.znear = 0.01  

        self.trans = trans
        self.scale = scale
        
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        if cyr != 0.0 :
            self.cxr = cxr
            self.cyr = cyr
            self.projection_matrix = getProjectionMatrixCV(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy, cx=cxr, cy=cyr).transpose(0,1).cuda()
        else:
            self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        if rayd is not None:
            projectinverse = self.projection_matrix.T.inverse()
            camera2wold = self.world_view_transform.T.inverse()
            pixgrid = create_meshgrid(self.image_height, self.image_width, normalized_coordinates=False, device="cpu")[0]
            pixgrid = pixgrid.cuda()  # H,W,
            
            xindx = pixgrid[:,:,0] # x 
            yindx = pixgrid[:,:,1] # y
      
            
            ndcy, ndcx = pix2ndc(yindx, self.image_height), pix2ndc(xindx, self.image_width)
            ndcx = ndcx.unsqueeze(-1)
            ndcy = ndcy.unsqueeze(-1)# * (-1.0)
            
            ndccamera = torch.cat((ndcx, ndcy,   torch.ones_like(ndcy) * (1.0) , torch.ones_like(ndcy)), 2) # N,4 

            projected = ndccamera @ projectinverse.T 
            diretioninlocal = projected / projected[:,:,3:] #v 


            direction = diretioninlocal[:,:,:3] @ camera2wold[:3,:3].T 
            rays_d = torch.nn.functional.normalize(direction, p=2.0, dim=-1)

            
            self.rayo = self.camera_center.expand(rays_d.shape).permute(2, 0, 1).unsqueeze(0)                                     #rayo.permute(2, 0, 1).unsqueeze(0)
            self.rayd = rays_d.permute(2, 0, 1).unsqueeze(0)    
            

        else :
            self.rayo = None
            self.rayd = None
            
    def load_dynamic_mask(self):
        if self.dynamic_mask_path is None:
            self.dynamic_mask = None
            return
        if self.img_wh is not None:
            mask_size = self.img_wh
        elif self.image_width is not None and self.image_height is not None:
            mask_size = (self.image_width, self.image_height)
        else:
            mask_size = None
        dynamic_mask = Image.open(self.dynamic_mask_path).convert("L")
        if mask_size is not None:
            dynamic_mask = dynamic_mask.resize(mask_size, Image.NEAREST)
        self.dynamic_mask = self.transform(dynamic_mask)[:1].clamp(0.0, 1.0)

    def load_motion_prior(self):
        if self.motion_prior_path is None:
            self.motion_prior = None
            return
        if self.img_wh is not None:
            prior_size = self.img_wh
        elif self.image_width is not None and self.image_height is not None:
            prior_size = (self.image_width, self.image_height)
        else:
            prior_size = None
        if self.motion_prior_path.endswith(".npz") or self.motion_prior_path.endswith(".npy"):
            data = np.load(self.motion_prior_path)
            if isinstance(data, np.lib.npyio.NpzFile):
                if "confidence" in data:
                    prior = data["confidence"].astype(np.float32)
                elif "motion" in data:
                    prior = data["motion"].astype(np.float32)
                elif "residual_mag" in data:
                    prior = data["residual_mag"].astype(np.float32)
                    finite = np.isfinite(prior)
                    if finite.any():
                        scale = np.percentile(prior[finite], 99.0)
                        prior = prior / max(float(scale), 1e-6)
                else:
                    first_key = list(data.keys())[0]
                    prior = data[first_key].astype(np.float32)
            else:
                prior = data.astype(np.float32)
            prior = np.nan_to_num(prior, nan=0.0, posinf=0.0, neginf=0.0)
            prior = torch.from_numpy(prior)
            if prior.dim() == 3:
                prior = prior[..., 0] if prior.shape[-1] <= 4 else prior[0]
            prior = prior.unsqueeze(0).unsqueeze(0).float()
            if prior_size is not None:
                prior = torch.nn.functional.interpolate(prior, size=(prior_size[1], prior_size[0]), mode="bilinear", align_corners=False)
            self.motion_prior = prior.squeeze(0).clamp(0.0, 1.0)
        else:
            prior_img = Image.open(self.motion_prior_path).convert("L")
            if prior_size is not None:
                prior_img = prior_img.resize(prior_size, Image.BILINEAR)
            self.motion_prior = self.transform(prior_img)[:1].clamp(0.0, 1.0)

    def load_depth_map(self):
        if self.depth_path is None:
            self.depth_map = None
            return
        import os
        if self.img_wh is not None:
            depth_size = self.img_wh
        elif self.image_width is not None and self.image_height is not None:
            depth_size = (self.image_width, self.image_height)
        else:
            depth_size = None
        if self.depth_path.endswith(".npy"):
            depth = np.load(self.depth_path).astype(np.float32)
            depth = torch.from_numpy(depth)
            if depth.dim() == 3:
                depth = depth.squeeze()
            depth = depth.unsqueeze(0).unsqueeze(0)
            if depth_size is not None:
                depth = torch.nn.functional.interpolate(depth, size=(depth_size[1], depth_size[0]), mode="bilinear", align_corners=False)
            self.depth_map = depth.squeeze(0).clamp_min(0.0)
        else:
            depth_img = Image.open(self.depth_path).convert("F")
            if depth_size is not None:
                depth_img = depth_img.resize(depth_size, Image.BILINEAR)
            self.depth_map = self.transform(depth_img)[:1].float().clamp_min(0.0)

    def load_image(self):
        original_image = Image.open(self.image_path).convert("RGB")
        original_image = original_image.resize(self.img_wh, Image.LANCZOS)
        self.original_image = self.transform(original_image)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
        if self.gt_alpha_mask is not None:
            self.original_image *= self.gt_alpha_mask.to(self.data_device)
        if self.dynamic_mask is None and self.dynamic_mask_path is not None:
            self.load_dynamic_mask()
        if self.motion_prior is None and self.motion_prior_path is not None:
            self.load_motion_prior()
        if self.depth_map is None and self.depth_path is not None:
            self.load_depth_map()
    
    def set_image(self):
        self.image_width = self.img_wh[0]
        self.image_height = self.img_wh[1]


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]


class Camerass(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", near=0.01, far=100.0, timestamp=0.0, rayo=None, rayd=None, rays=None, cxr=0.0,cyr=0.0,
                 ):
        super(Camerass, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.timestamp = timestamp
        self.fisheyemapper = None

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        # image is real image 
        if not isinstance(image, tuple):
            if "camera_" not in image_name:
                self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
            else:
                self.original_image = image.clamp(0.0, 1.0).half().to(self.data_device)
            print("read one")# lazy loader?
            self.image_width = self.original_image.shape[2]
            self.image_height = self.original_image.shape[1]

        else:
            self.image_width = image[0] 
            self.image_height = image[1] 
            self.original_image = None
        
        self.image_width = 2 * self.image_width
        self.image_height = 2 * self.image_height # 

        self.zfar = 100.0
        self.znear = 0.01  
        self.trans = trans
        self.scale = scale

        # w2c 
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        if cyr != 0.0 :
            self.cxr = cxr
            self.cyr = cyr
            self.projection_matrix = getProjectionMatrixCV(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy, cx=cxr, cy=cyr).transpose(0,1).cuda()
        else:
            self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


        if rayd is not None:
            projectinverse = self.projection_matrix.T.inverse()
            camera2wold = self.world_view_transform.T.inverse()
            pixgrid = create_meshgrid(self.image_height, self.image_width, normalized_coordinates=False, device="cpu")[0]
            pixgrid = pixgrid.cuda()  # H,W,
            
            xindx = pixgrid[:,:,0] # x 
            yindx = pixgrid[:,:,1] # y
      
            
            ndcy, ndcx = pix2ndc(yindx, self.image_height), pix2ndc(xindx, self.image_width)
            ndcx = ndcx.unsqueeze(-1)
            ndcy = ndcy.unsqueeze(-1)# * (-1.0)
            
            ndccamera = torch.cat((ndcx, ndcy,   torch.ones_like(ndcy) * (1.0) , torch.ones_like(ndcy)), 2) # N,4 

            projected = ndccamera @ projectinverse.T 
            diretioninlocal = projected / projected[:,:,3:] # 

            direction = diretioninlocal[:,:,:3] @ camera2wold[:3,:3].T 
            rays_d = torch.nn.functional.normalize(direction, p=2.0, dim=-1)

            
            self.rayo = self.camera_center.expand(rays_d.shape).permute(2, 0, 1).unsqueeze(0)                                     #rayo.permute(2, 0, 1).unsqueeze(0)
            self.rayd = rays_d.permute(2, 0, 1).unsqueeze(0)                                                                          #rayd.permute(2, 0, 1).unsqueeze(0)
        else :
            self.rayo = None
            self.rayd = None
