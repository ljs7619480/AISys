import argparse, os, yaml

import cv2
import math3d as m3d
import numpy as np
import open3d as o3d

homolize = lambda array: np.vstack((array, np.ones_like(array[0:1])))
dehomolize = lambda array: array[:-1]
bgr2rgb = lambda array: array[:,:,[2,1,0]]


class SourceManager:
    def __init__(self, dataset_path, meta_path):
        assert os.path.isdir(dataset_path)
        self.dataset_path = dataset_path
        self.pose = np.loadtxt(os.path.join(self.dataset_path, "GT_pose.txt"))
        with open(meta_path, 'r') as meta_file:
            meta = yaml.safe_load(meta_file)['cam_param']
        self.camK = np.array([[meta['fx'],          0, meta['cx']],
                              [         0, meta['fy'], meta['cy']],
                              [         0,          0,          1]])
        self.dscale = meta['dscale']

    def __getitem__(self, idx):
        path_template = os.path.join(self.dataset_path, 'images', '%03d-{}.png' %idx)
        color = cv2.imread(path_template.format('color'))
        depth = cv2.imread(path_template.format('depth'), cv2.IMREAD_ANYDEPTH).astype(np.uint16)
        # label = cv2.imread(path_template.format('label'), cv2.IMREAD_ANYDEPTH).astype(np.uint16)
        cam_pose = m3d.Transform()
        cam_pose.pos.set_array(self.pose[idx][:3])
        versor = m3d.Versor(*self.pose[idx][3:])
        cam_pose.orient.set_versor(versor)
        cam_pose.orient.rotateX(np.pi)
        camTworld = cam_pose.get_array()

        return color, depth, camTworld


    def __len__(self):
        return self.pose.shape[0]


def depth_image_to_point_cloud(color, depth, camK, cam_pose=None):
    width, height = color.shape[:2]
    u = np.tile(np.arange(0, width), height)
    v = np.repeat(np.arange(0, height), width)
    uv_homo = np.vstack((u, v, np.ones_like(u)))
    xyz = (np.linalg.inv(camK) @ uv_homo) * depth.reshape(-1)
    rgb = bgr2rgb(color).reshape(-1,3)
    mask_valid = np.where(depth.reshape(-1) > 0)[0]
    xyz = xyz[:, mask_valid]
    rgb = rgb[mask_valid]

    if cam_pose is not None:
        xyz_homo = homolize(xyz)
        xyz_homo = cam_pose @ xyz_homo
        xyz = dehomolize(xyz_homo)

    return xyz.T, rgb


def local_icp_algorithm(src_pts, tgt_pts, init_tran):
    """[summary]

        Args:
            src_pts ([type]): [description]
            tgt_pts ([type]): [description]
            init_tran ([type]): [description]
    """
    src_pts = homolize(src_pts)
    src_pts = init_tran @ src_pts
    src_pts = dehomolize(src_pts)

    mu_src = np.mean(src_pts, axis=1)
    mu_tgt = np.mean(tgt_pts, axis=1)
    W = src_pts @ tgt_pts.T
    u, s, vh = np.linalg.svd(W)
    rot = u @ vh
    vec = mu_src - (rot @ mu_tgt)

    tran = np.eye(4)
    tran[:3, :3] = rot
    tran[:3, 3:] = vec

    return tran


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_base', default="Data_collection", type=str)
    parser.add_argument('--sub_dir', default="first_floor", type=str)
    parser.add_argument('--sample_idx', type=int)
    
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    data_path = os.path.join(args.dataset_base, args.sub_dir)
    meta_path = os.path.join(args.dataset_base, "metadata.yaml")
    srcMng = SourceManager(data_path, meta_path)

    points, colors = [], []
    for i in range(len(srcMng)):
        color, depth, cam_pose = srcMng[i]
        depth = depth / srcMng.dscale
        xyz, rgb = depth_image_to_point_cloud(color, depth, srcMng.camK, cam_pose)
        points.append(xyz)
        colors.append(rgb)

    points = np.vstack(points)
    colors = np.vstack(colors).astype(float)/255

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd = pcd.voxel_down_sample(0.01)

    bbox = o3d.geometry.AxisAlignedBoundingBox(np.array([-50, -2, -50]).T, np.array([50, 0.6, 50]).T)
    pcd = pcd.crop(bbox)

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    o3d.visualization.draw_geometries([pcd, frame])

