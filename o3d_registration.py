import open3d as o3d

__all__ = ["global_ransac_registration", 
           "global_fast_registration",
           "local_icp_registration"]

def get_fpfh_feat(pcd, voxel_size):
    """http://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html#Extract-geometric-feature
    """
    radius_normal = voxel_size * 2
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )

    return pcd, pcd_fpfh


def global_ransac_registration(pcd_src, pcd_tgt, voxel_size):
    """http://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html#RANSAC
    """
    src_down, src_fpfh = get_fpfh_feat(pcd_src, voxel_size)
    tgt_down, tgt_fpfh = get_fpfh_feat(pcd_tgt, voxel_size)
    
    distance_threshold = voxel_size * 10
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source=src_down,
        target=tgt_down,
        source_feature=src_fpfh,
        target_feature=tgt_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
        #   o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
          o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria())
    
    return result.transformation


def global_fast_registration(pcd_src, pcd_tgt, voxel_size):
    """http://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html#Fast-global-registration
    """
    src_down, src_fpfh = get_fpfh_feat(pcd_src, voxel_size)
    tgt_down, tgt_fpfh = get_fpfh_feat(pcd_tgt, voxel_size)
    
    distance_threshold = voxel_size * 10
    result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
        src_down, tgt_down, src_fpfh, tgt_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    
    return result.transformation


def local_icp_registration(source, target, init_tran, distance_threshold, max_itr=2000):
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, init_tran,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_itr))

    return result.transformation