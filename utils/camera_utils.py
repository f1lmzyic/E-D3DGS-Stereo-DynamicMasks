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

from scene.cameras import Camera
from scene.cameras import Camerass # ass 
import numpy as np

from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
import torch 
import os 
WARNED = False
MISSING_MOTION_PRIORS = set()

def resolve_motion_prior_path(args, image_path):
    if not getattr(args, "use_motion_priors", False) or image_path is None:
        return None

    source_path = getattr(args, "source_path", "")
    images_dir = getattr(args, "images", "images")
    sidecar_dir = getattr(args, "motion_prior_dir", "motion_priors")
    images_root = images_dir if os.path.isabs(images_dir) else os.path.join(source_path, images_dir)
    sidecar_root = sidecar_dir if os.path.isabs(sidecar_dir) else os.path.join(source_path, sidecar_dir)

    try:
        rel_path = os.path.relpath(image_path, images_root)
    except ValueError:
        rel_path = os.path.basename(image_path)
    if rel_path.startswith(".."):
        rel_path = os.path.basename(image_path)

    candidate = os.path.join(sidecar_root, rel_path)
    stem, _ = os.path.splitext(candidate)
    # Prefer rich NPZ/NPY sidecars over the inspection PNG written by the generator.
    candidates = [stem + ext for ext in [".npz", ".npy", ".png", ".jpg", ".jpeg"]]
    candidates.append(candidate)
    for path in candidates:
        if os.path.exists(path):
            return path

    if image_path not in MISSING_MOTION_PRIORS and len(MISSING_MOTION_PRIORS) < 20:
        print(f"[WARN] Motion prior not found for {image_path}; expected under {sidecar_root}")
        MISSING_MOTION_PRIORS.add(image_path)
    return None


def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    cameradirect = cam_info.hpdirecitons
    camerapose = cam_info.pose 
     
    if camerapose is not None:
        c2w = torch.FloatTensor(camerapose)
        rays_o, rays_d = get_image_rays(cameradirect, c2w)# get_image_rays          get_rays
    else :
        rays_o = None
        rays_d = None
    motion_prior_path = resolve_motion_prior_path(args, cam_info.image_path)
    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device, near=cam_info.near, far=cam_info.far, timestamp=cam_info.timestamp, rayo=rays_o, rayd=rays_d,
                  motion_prior_path=motion_prior_path)



def loadCamv2(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.width, cam_info.height

    if args.resolution in [1., 2., 4., 8.]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    # print(orig_w, orig_h)
    # # print(scale)
    # print(resolution, args.resolution, resolution_scale)
    # exit()

    if cam_info.image:
        resized_image_rgb = PILtoTorch(cam_info.image, resolution)
        gt_image = resized_image_rgb[:3, ...]
        loaded_mask = None
        if resized_image_rgb.shape[1] == 4:
            loaded_mask = resized_image_rgb[3:4, ...]
    else:
        resized_image_rgb = None
        gt_image = None
        loaded_mask = None

    cameradirect = cam_info.hpdirecitons
    camerapose = cam_info.pose 
    if cam_info.image_name is not None:
        frame_stem = os.path.splitext(os.path.basename(cam_info.image_name))[0]
        frame_no = int(frame_stem) if frame_stem.isdigit() else cam_info.uid
    else:
        frame_no = cam_info.uid
    camera_dir = os.path.basename(os.path.dirname(cam_info.image_path)) if cam_info.image_path is not None else ""
    if camera_dir.startswith("cam") and camera_dir[3:].isdigit():
        cam_no = int(camera_dir[3:])
    elif camera_dir.lower() == "left":
        cam_no = 0
    elif camera_dir.lower() == "right":
        cam_no = 1
    else:
        cam_no = cam_info.uid

    if camerapose is not None:
        rays_o, rays_d = 1, cameradirect
    else :
        rays_o = None
        rays_d = None

    motion_prior_path = resolve_motion_prior_path(args, cam_info.image_path)
    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device='cpu', near=cam_info.near, far=cam_info.far, timestamp=cam_info.timestamp, rayo=rays_o, rayd=rays_d,cxr=cam_info.cxr,cyr=cam_info.cyr,
                  cam_no=cam_no, frame_no=frame_no, image_path=cam_info.image_path, img_wh=resolution,
                  motion_prior_path=motion_prior_path)


def loadCamHyper(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.width, cam_info.height

    if args.resolution in [1., 2., 4., 8.]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    if cam_info.image:
        resized_image_rgb = PILtoTorch(cam_info.image, resolution)
        gt_image = resized_image_rgb[:3, ...]
        loaded_mask = None
        if resized_image_rgb.shape[1] == 4:
            loaded_mask = resized_image_rgb[3:4, ...]
    else:
        resized_image_rgb = None
        gt_image = None
        loaded_mask = None

    cameradirect = cam_info.hpdirecitons
    camerapose = cam_info.pose 
     
    # if camerapose is not None and cam_info.image is not None:
    #     rays_o, rays_d = 1, cameradirect
    # else :
    rays_o = None
    rays_d = None

    cam_no = 0 
    frame_no = int(cam_info.image_name.split('.')[-2].split('_')[-1])

    image_path = getattr(cam_info, "image_path", None)
    motion_prior_path = resolve_motion_prior_path(args, image_path)
    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device='cpu', near=cam_info.near, far=cam_info.far, timestamp=cam_info.timestamp, rayo=rays_o, rayd=rays_d,cxr=cam_info.cxr,cyr=cam_info.cyr,
                  cam_no=cam_no, frame_no=frame_no, image_path=image_path, img_wh=resolution,
                  motion_prior_path=motion_prior_path)







def loadCamv2timing(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    cameradirect = cam_info.hpdirecitons
    camerapose = cam_info.pose 
     
    if camerapose is not None:
        rays_o, rays_d = 1, cameradirect
    else :
        rays_o = None
        rays_d = None

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device, near=cam_info.near, far=cam_info.far, timestamp=cam_info.timestamp, rayo=rays_o, rayd=rays_d,cxr=cam_info.cxr,cyr=cam_info.cyr)



def loadCamv2ss(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size
    assert args.resolution == 1
    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resolution = (int(orig_w / 2), int(orig_h / 2))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution) # hard coded half resolution

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    cameradirect = cam_info.hpdirecitons
    camerapose = cam_info.pose 
     
    if camerapose is not None:
        rays_o, rays_d = 1, cameradirect
    else :
        rays_o = None
        rays_d = None
    return Camerass(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device, near=cam_info.near, far=cam_info.far, timestamp=cam_info.timestamp, rayo=rays_o, rayd=rays_d,cxr=cam_info.cxr,cyr=cam_info.cyr)



def loadCamnogt(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.width, cam_info.height

    if args.resolution in [1., 2., 4., 8.]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = None #PILtoTorch(cam_info.image, resolution)

    gt_image = None #resized_image_rgb[:3, ...]
    loaded_mask = None

    loaded_mask = None

    cameradirect = cam_info.hpdirecitons
    camerapose = cam_info.pose 
     
    rays_o, rays_d = 1, cameradirect

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=resolution, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device, near=cam_info.near, far=cam_info.far, timestamp=cam_info.timestamp, rayo=rays_o, rayd=rays_d)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def cameraList_from_camInfosv2(cam_infos, resolution_scale, args, ss=False):
    camera_list = []
    from tqdm import tqdm

    if not ss: #
        for id, c in tqdm(enumerate(cam_infos), total=len(cam_infos)):
            camera_list.append(loadCamv2(args, id, c, resolution_scale))
    else:
        for id, c in tqdm(enumerate(cam_infos), total=len(cam_infos)):
            camera_list.append(loadCamv2ss(args, id, c, resolution_scale))

    return camera_list


def cameraList_from_camInfosHyper(cam_infos, resolution_scale, args):
    camera_list = []
    from tqdm import tqdm

    for id, c in tqdm(enumerate(cam_infos), total=len(cam_infos)):
        camera_list.append(loadCamHyper(args, id, c, resolution_scale))



    return camera_list
def cameraList_from_camInfosv2nogt(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCamnogt(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
