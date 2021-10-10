import argparse, os, yaml

import cv2
import math3d as m3d
import numpy as np
import open3d as o3d
from tqdm import tqdm

from o3d_registration import global_ransac_registration, local_icp_registration


homolize = lambda array: np.vstack((array, np.ones_like(array[0:1])))
dehomolize = lambda array: array[:-1]
transform = lambda pose, xyz: dehomolize(pose @ homolize(xyz))
bgr2rgb = lambda array: array[:,:,[2,1,0]]


class SourceManager:
    """A simple Data_collection loader
    """
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
        cam_pose.orient.rotateX(np.pi) # Note: given rotation poses is not relative to word coordinate.
        camTworld = cam_pose.get_array()

        return color, depth, camTworld


    def __len__(self):
        return self.pose.shape[0]


def depth_image_to_point_cloud(color, depth, camK):
    """ Reconstruct point cloud from rgbd images

        Args:
            color (float[HxWx3]): RGB image 
            depth (float[HxWx1]): Depth image
            camK (float[3x3]): Pinhole camera intrinsic matrix

        Returns:
            xyz (float[3x(HxW)]): points
            rgb (float[3x(HxW)]): point colors 
    """
    width, height = color.shape[:2]
    u = np.tile(np.arange(0, width), height)
    v = np.repeat(np.arange(0, height), width)
    uv_homo = np.vstack((u, v, np.ones_like(u)))
    xyz = (np.linalg.inv(camK) @ uv_homo) * depth.reshape(-1)
    rgb = bgr2rgb(color).reshape(-1,3) / 255
    mask_valid = np.where(depth.reshape(-1) > 0)[0]
    xyz = xyz[:, mask_valid]
    rgb = rgb[mask_valid]

    return xyz, rgb


# To Be Done
# def __local_icp_algorithm(src_pts, tgt_pts, init_tran):
#     src_pts = homolize(src_pts)
#     src_pts = init_tran @ src_pts
#     src_pts = dehomolize(src_pts)

#     mu_src = np.mean(src_pts, axis=1)
#     mu_tgt = np.mean(tgt_pts, axis=1)
#     W = src_pts @ tgt_pts.T
#     u, s, vh = np.linalg.svd(W)
#     rot = u @ vh
#     vec = mu_src - (rot @ mu_tgt)

#     tran = np.eye(4)
#     tran[:3, :3] = rot
#     tran[:3, 3:] = vec

#     return tran


def estimate_transform(pcd_src, pcd_tgt, voxel_size):
    pcd_src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_src.T))
    pcd_tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_tgt.T))
    pcd_src = pcd_src.voxel_down_sample(0.03)
    pcd_tgt = pcd_tgt.voxel_down_sample(0.03)
    
    pose = global_ransac_registration(pcd_src, pcd_tgt, voxel_size)
    transform = local_icp_registration(pcd_src, pcd_tgt, pose, voxel_size*10)
    
    return transform


def pose2lineset(poses, color, y=None):
    """ Convert transformation matrix to LineSet object
        Extract the translation vector from matries and return a lineset object
        that present the moving trajectory.

        Args:
            poses (float[Nx4x4]): tranformation matries
            color (float[1x3]): line color (0 <= r,g,b <= 1)
            y     (float, optional): set trajector height to constant value. 
                    Defaults to None.

        Returns:
            [o3d.geometry.LineSet]: 
    """
    points = np.hstack([pose[:3, -1:] for pose in poses]).T # Nx3
    if y is not None:
        points[:, 1:2] = np.full_like(points[:, 1:2], y)
    lines = np.arange(len(poses)-1)
    lines = np.vstack((lines, lines+1)).T
    lineset = o3d.geometry.LineSet(
        o3d.utility.Vector3dVector(points),
        o3d.utility.Vector2iVector(lines),
    )
    lineset = lineset.paint_uniform_color(np.array(color, dtype=float).T)

    return lineset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('sub_dir', type=str)
    parser.add_argument('--dataset_base', default="Data_collection", type=str)
    parser.add_argument('--sample_idx', type=int)
    parser.add_argument('--keep_roof', action="store_true")
    parser.add_argument('--voxel_size', default=0.03, type=float)
    parser.add_argument('--cache_pcd', action="store_true", type=bool)
    
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    data_path = os.path.join(args.dataset_base, args.sub_dir)
    meta_path = os.path.join(args.dataset_base, "metadata.yaml")
    srcMng = SourceManager(data_path, meta_path)

    points, colors, gt_pose, esti_pose = [], [], [], []
    for i in tqdm(range(len(srcMng)), desc="Estimating poses", ncols=80):
        color, depth, cam_pose = srcMng[i]
        depth = depth / srcMng.dscale
        # point cloud
        xyz, rgb = depth_image_to_point_cloud(color, depth, srcMng.camK)
        points.append(xyz)
        colors.append(rgb)
        # pose
        gt_pose.append(cam_pose)
        if not (i == 0):
            xyz_prev = points[i-1]
            pose = estimate_transform(xyz, xyz_prev, args.voxel_size)
            esti_pose.append(esti_pose[i-1] @ pose)
        else: # init_frame
            esti_pose.append(cam_pose)

    # reconstruct point cloud and cache point cloud
    cache_path = args.sub_dir+'.pts'
    if os.path.exists(cache_path):
        pcd = o3d.io.read_point_cloud(cache_path)
    else:
        for i in tqdm(range(len(points)), desc="Transform points", ncols=80):
            points[i] = transform(gt_pose[i], points[i])
        points = np.hstack(points).T
        # points = np.vstack([transform(pose, xyz).T for pose, xyz in zip(gt_pose, points)])
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        pcd.colors = o3d.utility.Vector3dVector(np.vstack(colors))
        if args.cache_pcd:
            o3d.io.write_point_cloud(cache_path, pcd)
    
    # crop roof
    pcd = pcd.voxel_down_sample(args.voxel_size)
    if not args.keep_roof:
        bbox = o3d.geometry.AxisAlignedBoundingBox(np.array([-50, -2, -50]).T, np.array([50, 0.6, 50]).T)
        pcd = pcd.crop(bbox)

    gt_path = pose2lineset(gt_pose, color=[0, 0, 0], y=0)
    esti_path = pose2lineset(esti_pose, color=[1, 0, 0], y=0)

    # Visualize
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    o3d.visualization.draw_geometries([pcd, frame, gt_path, esti_path])
