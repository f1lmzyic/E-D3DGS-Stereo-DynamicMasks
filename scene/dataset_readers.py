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

import os
import sys
import re
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from scene.hyper_loader import Load_hyper_data, format_hyper_data
import copy
from utils.graphics_utils import getWorld2View2, focal2fov
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from utils.graphics_utils import BasicPointCloud
import glob
import natsort
import torch
from tqdm import tqdm


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    near: float
    far: float
    timestamp: float
    pose: np.array 
    hpdirecitons: np.array
    cxr: float
    cyr: float

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    video_cameras: list
    nerf_normalization: dict
    ply_path: str
    

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def _get_fov_from_intrinsics(intr):
    if intr.model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"]:
        focal_length_x = intr.params[0]
        focal_length_y = intr.params[0]
    elif intr.model in ["PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "THIN_PRISM_FISHEYE"]:
        focal_length_x = intr.params[0]
        focal_length_y = intr.params[1]
    else:
        raise AssertionError(f"Colmap camera model not handled: {intr.model}")
    FovY = focal2fov(focal_length_y, intr.height)
    FovX = focal2fov(focal_length_x, intr.width)
    return FovY, FovX


def _resolve_sparse_dir(path):
    candidates = [
        os.path.join(path, "colmap/dense/workspace/sparse"),
        os.path.join(path, "ns_output/colmap/sparse/0"),
        os.path.join(path, "ns_output/colmap/sparse_work/0"),
        os.path.join(path, "ns_output/colmap/sparse"),
    ]
    for candidate in candidates:
        has_images = os.path.exists(os.path.join(candidate, "images.bin")) or os.path.exists(os.path.join(candidate, "images.txt"))
        has_cameras = os.path.exists(os.path.join(candidate, "cameras.bin")) or os.path.exists(os.path.join(candidate, "cameras.txt"))
        if has_images and has_cameras:
            return candidate
    return os.path.join(path, "colmap/dense/workspace/sparse")


def _resolve_images_root(path, images):
    if images is None or images == "":
        images = "images"
    if os.path.isabs(images):
        return images
    return os.path.join(path, images)


def _build_image_lookup(images_root):
    lookup = {}
    for root, dirs, files in os.walk(images_root):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d.lower() != ".ipynb_checkpoints"]
        for file_name in files:
            ext = os.path.splitext(file_name)[1].lower()
            if ext not in [".png", ".jpg", ".jpeg"]:
                continue
            if Path(file_name).stem.lower().endswith("-checkpoint"):
                continue
            full_path = os.path.join(root, file_name)
            lookup[file_name] = full_path
    return lookup


def _build_image_sequence_lookup(images_root):
    by_view = {}
    for root, dirs, files in os.walk(images_root):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d.lower() != ".ipynb_checkpoints"]
        rel_parent = os.path.relpath(root, images_root)
        parent_view = "" if rel_parent == "." else os.path.basename(rel_parent)
        for file_name in files:
            ext = os.path.splitext(file_name)[1].lower()
            if ext not in [".png", ".jpg", ".jpeg"]:
                continue
            if Path(file_name).stem.lower().endswith("-checkpoint"):
                continue
            parsed_view, frame = _parse_view_and_frame(file_name)
            if frame is None:
                continue
            view = parent_view if parent_view else parsed_view
            by_view.setdefault(view, {})[frame] = os.path.join(root, file_name)
    return by_view


def _nearest_frame_record(frame_map, frame):
    if frame in frame_map:
        return frame_map[frame]
    if not frame_map:
        return None
    nearest = min(frame_map.keys(), key=lambda k: abs(k - frame))
    return frame_map[nearest]


def _parse_view_and_frame(file_name):
    stem = Path(file_name).stem
    match = re.match(r"^(.*?)(\d+)$", stem)
    if match:
        view = match.group(1) if match.group(1) else "cam"
        frame = int(match.group(2))
    else:
        view = stem
        frame = None
    return view, frame


def _is_ignored_colmap_image_name(image_name):
    p = Path(image_name)
    if any(part.startswith(".") or part.lower() == ".ipynb_checkpoints" for part in p.parts):
        return True
    if p.stem.lower().endswith("-checkpoint"):
        return True
    return False


def _resolve_point_cloud_path(path, sparse_dir):
    ply_candidates = [
        os.path.join(path, "points3D_downsample.ply"),
        os.path.join(path, "points3D.ply"),
        os.path.join(sparse_dir, "points3D.ply"),
        os.path.join(path, "colmap/dense/workspace/fused.ply"),
    ]
    for candidate in ply_candidates:
        if os.path.exists(candidate):
            return candidate

    bin_candidates = [
        os.path.join(sparse_dir, "points3D.bin"),
        os.path.join(path, "ns_output/colmap/sparse/0/points3D.bin"),
        os.path.join(path, "ns_output/colmap/sparse_work/0/points3D.bin"),
    ]
    txt_candidates = [
        os.path.join(sparse_dir, "points3D.txt"),
        os.path.join(path, "ns_output/colmap/sparse/0/points3D.txt"),
        os.path.join(path, "ns_output/colmap/sparse_work/0/points3D.txt"),
    ]
    generated_ply = os.path.join(path, "points3D_downsample.ply")

    for candidate in bin_candidates:
        if os.path.exists(candidate):
            xyz, rgb, _ = read_points3D_binary(candidate)
            storePly(generated_ply, xyz, np.clip(rgb, 0, 255).astype(np.uint8))
            return generated_ply
    for candidate in txt_candidates:
        if os.path.exists(candidate):
            xyz, rgb, _ = read_points3D_text(candidate)
            storePly(generated_ply, xyz, np.clip(rgb, 0, 255).astype(np.uint8))
            return generated_ply

    return generated_ply


def _camera_view_key(image_name):
    image_stem = os.path.splitext(os.path.basename(image_name))[0]
    image_dir = os.path.basename(os.path.dirname(image_name))
    if image_dir and image_dir != ".":
        return image_dir
    if image_stem.startswith("cam") and len(image_stem) >= 5:
        return image_stem[:5]
    parsed = re.match(r"^(.*?)(\d+)$", image_stem)
    if parsed and parsed.group(1):
        return parsed.group(1)
    return image_stem


def readColmapCamerasDynerf(cam_extrinsics, cam_intrinsics, images_folder, near, far, startime=0, duration=300, images_subdir="images"):
    cam_infos = []
    keys = sorted(cam_extrinsics.keys())
    if len(keys) == 0:
        return cam_infos

    images_root = _resolve_images_root(images_folder, images_subdir)
    first_name = Path(cam_extrinsics[keys[0]].name).stem
    folder_mode = os.path.exists(os.path.join(images_root, first_name, f"{startime:04d}.png"))

    if folder_mode:
        for idx, key in enumerate(keys):
            sys.stdout.write('\r')
            sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
            sys.stdout.flush()

            extr = cam_extrinsics[key]
            if _is_ignored_colmap_image_name(extr.name):
                continue
            intr = cam_intrinsics[extr.camera_id]
            height = intr.height
            width = intr.width

            uid = intr.id
            R = np.transpose(qvec2rotmat(extr.qvec))
            T = np.array(extr.tvec)
            FovY, FovX = _get_fov_from_intrinsics(intr)

            for j in range(startime, startime + int(duration)):
                image_path = os.path.join(images_root, f"{Path(extr.name).stem}", "%04d.png" % j)
                image_name = os.path.join(f"{Path(extr.name).stem}", image_path.split('/')[-1])

                if not os.path.exists(image_path):
                    continue
                if j == startime:
                    image = Image.open(image_path).convert("RGB")
                    image = image.resize((int(width), int(height)), Image.LANCZOS)
                    cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path, image_name=image_name, width=width, height=height, near=near, far=far, timestamp=(j-startime)/max(int(duration), 1), pose=1, hpdirecitons=1,cxr=0.0, cyr=0.0)
                else:
                    image = None
                    cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path, image_name=image_name, width=width, height=height, near=near, far=far, timestamp=(j-startime)/max(int(duration), 1), pose=None, hpdirecitons=None, cxr=0.0, cyr=0.0)
                cam_infos.append(cam_info)
        sys.stdout.write('\n')
        return cam_infos

    image_lookup = _build_image_lookup(images_root)
    image_sequences = _build_image_sequence_lookup(images_root)
    parsed = []
    for idx, key in enumerate(keys):
        image_name = cam_extrinsics[key].name
        if _is_ignored_colmap_image_name(image_name):
            continue
        view_name, frame_idx = _parse_view_and_frame(image_name)
        parsed.append((idx, key, view_name, frame_idx))

    valid_frames = [frame for _, _, _, frame in parsed if frame is not None]
    frame_base = min(valid_frames) if len(valid_frames) > 0 else 0
    target_duration = int(duration) if duration is not None else None
    if target_duration is None or target_duration <= 0:
        target_duration = max(len(parsed), 1)

    records_by_view = {}
    for idx, key, view_name, frame_idx in parsed:
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        image_basename = os.path.basename(extr.name)
        image_path = image_lookup.get(image_basename, None)
        if image_path is None:
            continue
        if frame_idx is None:
            frame_number = idx
        else:
            frame_number = max(frame_idx - frame_base, 0)
        records_by_view.setdefault(view_name, {})[frame_number] = (extr, intr, image_path)

    view_to_uid = {view_name: uid for uid, view_name in enumerate(sorted(records_by_view.keys()))}
    for view_idx, view_name in enumerate(sorted(records_by_view.keys())):
        frame_map = records_by_view[view_name]
        if len(frame_map) == 0:
            continue
        image_frame_map = image_sequences.get(view_name, {})
        for j in range(startime, startime + target_duration):
            sys.stdout.write('\r')
            sys.stdout.write("Reading camera {}/{}".format(view_idx+1, len(records_by_view)))
            sys.stdout.flush()

            record = _nearest_frame_record(frame_map, j)
            if record is None:
                continue
            extr, intr, image_path = record
            image_path = image_frame_map.get(j, image_frame_map.get(j + frame_base, image_path))
            height = intr.height
            width = intr.width
            uid = view_to_uid[view_name]
            R = np.transpose(qvec2rotmat(extr.qvec))
            T = np.array(extr.tvec)
            FovY, FovX = _get_fov_from_intrinsics(intr)

            frame_number = j - startime
            image_name = os.path.join(view_name, f"{frame_number:04d}.png")
            timestamp = frame_number / max(target_duration - 1, 1)

            if j == startime:
                image = Image.open(image_path).convert("RGB")
                image = image.resize((int(width), int(height)), Image.LANCZOS)
                cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path, image_name=image_name, width=width, height=height, near=near, far=far, timestamp=timestamp, pose=1, hpdirecitons=1,cxr=0.0, cyr=0.0)
            else:
                image = None
                cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path, image_name=image_name, width=width, height=height, near=near, far=far, timestamp=timestamp, pose=None, hpdirecitons=None, cxr=0.0, cyr=0.0)
            cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def readColmapCamerasTechnicolorTestonly(cam_extrinsics, cam_intrinsics, images_folder, near, far, startime=0, duration=None):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics): 
        sys.stdout.write('\r')
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        for j in range(startime, startime+ int(duration)):
            image_path = os.path.join(images_folder,f"images/{extr.name[:-4]}", "%04d.png" % j)
            image_name = os.path.join(f"{extr.name[:-4]}", image_path.split('/')[-1])
        
            cxr =   ((intr.params[2] )/  width - 0.5) 
            cyr =   ((intr.params[3] ) / height - 0.5) 

            assert os.path.exists(image_path), "Image {} does not exist!".format(image_path)
            
            if image_name == "cam10":
                image = Image.open(image_path).convert("RGB")
            else:
                image = None 

            if j == startime:
                cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path, image_name=image_name, width=width, height=height, near=near, far=far, timestamp=(j-startime)/duration, pose=1, hpdirecitons=1, cxr=cxr, cyr=cyr)
            else:
                cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path, image_name=image_name, width=width, height=height, near=near, far=far, timestamp=(j-startime)/duration, pose=None, hpdirecitons=None,  cxr=cxr, cyr=cyr)
            cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def readColmapCamerasTechnicolor(cam_extrinsics, cam_intrinsics, images_folder, near, far, startime=0, duration=None):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics): 
        sys.stdout.write('\r')
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
        for j in range(startime, startime+ int(duration)):
            image_path = os.path.join(images_folder,f"images/{extr.name[:-4]}", "%04d.png" % j)
            image_name = os.path.join(f"{extr.name[:-4]}", image_path.split('/')[-1])

            cxr =   ((intr.params[2] )/  width - 0.5) 
            cyr =   ((intr.params[3] ) / height - 0.5) 
    
            assert os.path.exists(image_path), "Image {} does not exist!".format(image_path)
            image = Image.open(image_path).convert("RGB")

            if j == startime:
                cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path, image_name=image_name, width=width, height=height, near=near, far=far, timestamp=(j-startime)/duration, pose=1, hpdirecitons=1, cxr=cxr, cyr=cyr)
            else:
                cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path, image_name=image_name, width=width, height=height, near=near, far=far, timestamp=(j-startime)/duration, pose=None, hpdirecitons=None,  cxr=cxr, cyr=cyr)
            cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def normalize(v):
    return v / np.linalg.norm(v)


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), #('t','f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readColmapSceneInfoDynerf(path, images, eval, duration=300, testonly=None):
    sparse_dir = _resolve_sparse_dir(path)
    try:
        cameras_extrinsic_file = os.path.join(sparse_dir, "images.bin")
        cameras_intrinsic_file = os.path.join(sparse_dir, "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(sparse_dir, "images.txt")
        cameras_intrinsic_file = os.path.join(sparse_dir, "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    near = 0.01
    far = 100

    cam_infos_unsorted = readColmapCamerasDynerf(
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        images_folder=path,
        near=near,
        far=far,
        duration=duration,
        images_subdir=images
    )
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    video_cam_infos = getSpiralColmap(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,near=near, far=far)

    view_keys = sorted(list(set(_camera_view_key(cam.image_name) for cam in cam_infos)))
    if len(view_keys) > 1:
        if "cam00" in view_keys:
            test_view = "cam00"
        elif "right" in view_keys and "left" in view_keys:
            test_view = "right"
        else:
            test_view = view_keys[0]
        train_cam_infos = [_ for _ in cam_infos if _camera_view_key(_.image_name) != test_view]
        test_cam_infos = [_ for _ in cam_infos if _camera_view_key(_.image_name) == test_view]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = [cam for idx, cam in enumerate(cam_infos) if idx % 8 == 0]
        if len(test_cam_infos) == 0 and len(train_cam_infos) > 0:
            test_cam_infos = train_cam_infos[:1]

    nerf_normalization = getNerfppNorm(train_cam_infos)
    ply_path = _resolve_point_cloud_path(path, sparse_dir)
    
    if not testonly:
        try:
            pcd = fetchPly(ply_path)
        except Exception as e:
            print("error:", e)
            pcd = None
    else:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=video_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readColmapSceneInfoTechnicolor(path, images, eval, duration=None, testonly=None):
    try:
        cameras_extrinsic_file = os.path.join(path, "colmap/dense/workspace/sparse", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "colmap/dense/workspace/sparse", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "colmap/dense/workspace/sparse", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "colmap/dense/workspace/sparse", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    near = 0.01
    far = 100

    if testonly:
        cam_infos_unsorted = readColmapCamerasTechnicolorTestonly(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=path, near=near, far=far, duration=duration)
    else:
        cam_infos_unsorted = readColmapCamerasTechnicolor(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=path, near=near, far=far, duration=duration)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
     
    train_cam_infos = [_ for _ in cam_infos if "cam10" not in _.image_name]
    test_cam_infos = [_ for _ in cam_infos if "cam10" in _.image_name]

    uniquecheck = []
    for cam_info in test_cam_infos:
        if cam_info.image_name[:5] not in uniquecheck:
            uniquecheck.append(cam_info.image_name[:5])
    assert len(uniquecheck) == 1 
    
    sanitycheck = []
    for cam_info in train_cam_infos:
        if  cam_info.image_name[:5] not in sanitycheck:
            sanitycheck.append( cam_info.image_name[:5])
    for testname in uniquecheck:
        assert testname not in sanitycheck

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3D_downsample.ply")
    if not testonly:
        try:
            pcd = fetchPly(ply_path)
        except:
            pcd = None
    else:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=[],
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readHyperDataInfos(datadir,use_bg_points, eval, startime=0, duration=None):
    train_cam_infos = Load_hyper_data(datadir, 0.5, use_bg_points, split ="train", startime=startime, duration=duration)
    test_cam_infos = Load_hyper_data(datadir, 0.5, use_bg_points, split="test", startime=startime, duration=duration)
    print("load finished")
    train_cam = format_hyper_data(train_cam_infos,"train", 
                                  near=train_cam_infos.near, far=train_cam_infos.far,
                                  startime=train_cam_infos.startime, duration=train_cam_infos.duration)
    print("format finished")
    video_cam_infos = copy.deepcopy(test_cam_infos)
    video_cam_infos.split="video"

    nerf_normalization = getNerfppNorm(train_cam)

    ply_path = os.path.join(datadir, "points3D_downsample.ply")
    pcd = fetchPly(ply_path)
    xyz = np.array(pcd.points)
    pcd = pcd._replace(points=xyz)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=video_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           )
    return scene_info


sceneLoadTypeCallbacks = {
    "Technicolor": readColmapSceneInfoTechnicolor,
    "Nerfies": readHyperDataInfos,
    "Dynerf": readColmapSceneInfoDynerf,
}

# modify the code in https://github.com/hustvl/4DGaussians/blob/master/scene/neural_3D_dataset_NDC.py
def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.eye(4)
    m[:3] = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def render_path_spiral(c2w, up, rads, zrate, N_rots=2, N=120):
    render_poses = []

    for theta in np.linspace(0.0, 2.0 * np.pi * N_rots, N + 1)[:-1]:
        d = np.dot(
            c2w[:3,:3],
            np.array([np.cos(theta), np.sin(theta), 1.]) * rads
        )
        c = c2w[:3,3] + d
        z = normalize(zrate * c2w[:3,2] - d)
        render_poses.append(viewmatrix(z, up, c))
    return render_poses

def get_spiral(c2ws_all, near, far, rads_scale=0.25, N_views=120):
    """
    Generate a set of poses using spiral camera trajectory as validation poses.
    """

    # test cam is the center
    c2w = c2ws_all[0,:3,:] 
    up = c2ws_all[0, :3, 1]

    # Find a reasonable "focus depth" for this dataset
    dt = 0.75
    zrate = (1.0 - dt) * (near + far)

    # Get radii for spiral path
    tt = c2ws_all[1:, :3, 3] - c2ws_all[0:1, :3, 3]
    rads = np.percentile(np.abs(tt), 90, 0) * rads_scale

    render_poses = render_path_spiral(
        c2w, up, rads, zrate, N_rots=3, N=N_views
    )
    return np.stack(render_poses)


def getSpiralColmap(cam_extrinsics, cam_intrinsics, near, far):
    c2ws_all = {}
    for idx, key in enumerate(cam_extrinsics): 
        sys.stdout.write('\r')
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        w2c = np.eye(4)
        w2c[:3,:3] = qvec2rotmat(extr.qvec)
        w2c[:3,3] = np.array(extr.tvec)
        c2w = np.linalg.inv(w2c)
        c2ws_all[key] = c2w[:3,:]
    c2ws_all = np.stack([value for _, value in sorted(c2ws_all.items())])

    if intr.model=="SIMPLE_PINHOLE":
        focal_length_x = intr.params[0]
        FovY = focal2fov(focal_length_x, height)
        FovX = focal2fov(focal_length_x, width)
    elif intr.model=="PINHOLE":
        focal_length_x = intr.params[0]
        focal_length_y = intr.params[1] 
        FovY = focal2fov(focal_length_y, height)
        FovX = focal2fov(focal_length_x, width)
    else:
        assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

    height = intr.height
    width = intr.width
    cam_infos = []
    render_poses = get_spiral(c2ws_all,near,far,N_views=300)

    for i,c2w in enumerate(render_poses):
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]
        image = None
        cam_info = CameraInfo(uid=i, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=None, image_name=None, width=width, height=height, near=near, far=far, timestamp=i/(len(render_poses) - 1), pose=None, hpdirecitons=None, cxr=0.0, cyr=0.0)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos
