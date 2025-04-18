import os
import open3d as o3d
import numpy as np
import robust_laplacian
import scipy.sparse.linalg as sla
import random
from typing import Optional


def fps_sample_numpy_point_cloud(point_cloud: np.array, sample_num_points: int) -> np.array:
    """
    Description
    -----------
    Samples sample_num_points from point_cloud using FPS algorithm.
    If sample_num_points > len(point_cloud), then return the original point_cloud.

    Parameters
    ----------
    point_cloud: np.array
        Point cloud, from which the nearest neighbors will be selected.
        This is in n x 3 numpy array.
    sample_num_points: int
        Number of points to sample.
        
    Returns
    -------
    np.array
        Returns sample_num_points x 3 numpy array if sample_num_points <= len(point_cloud).
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    if sample_num_points <= len(point_cloud):
        pcd.farthest_point_down_sample(sample_num_points)
    else: 
        print("Error: Sampled number of points is larger than the original!")
    return np.asarray(pcd.points)

def read_point_cloud(file_name: str,
                     is_mesh: bool=True,
                     use_vertices: bool=True,
                     decimate: bool=False,
                     decimate_to: int=12000,
                     sample_points: bool=True,
                     sampling_method: str='fps',
                     sample_num_points: int=6000,) -> np.array:
    """
    Description
    -----------
    Reads a 3D object file and returns a point cloud as a N x 3 numpy array format.

    Current working directory should be meshDL project folder.

    Parameters
    ----------
    file_name: str
        Should be obj/stl (if mesh) or pcd (if point cloud) file.
        Must be full path to the file.
    is_mesh: bool
        True if the 3D object file contains a mesh. False if the file contains point cloud.
    use_vertices: bool
        True if the returned point cloud data should be a subset of mesh vertices.
    decimate: bool
        True if the mesh should be decimated. False otherwise.
    decimate_to: int
        Target number of triangles to decimate the original mesh to.
    sample_points: bool
        True if the obtained point cloud should be downsampled.
        If False and the original data is mesh, the obtained point cloud consists of mesh vertices.
    sampling_method: str
        Point cloud sampling strategy. Options are 'fps', 'uniform', 'poisson disk'
    sample_num_points: int
        Number of points to sample.

    Returns
    -------
    np.array
        Point cloud data structure implemented as N x 3 numpy array format
    """

    point_cloud = o3d.geometry.PointCloud()

    if is_mesh:
        mesh = o3d.io.read_triangle_mesh(file_name)
        if decimate:
            mesh = mesh.simplify_quadric_decimation(decimate_to)
        if sample_points:
            if sampling_method == 'uniform':
                point_cloud = mesh.sample_points_uniformly(sample_num_points)
            else:
                pcd = None
                if use_vertices and len(mesh.vertices) > sample_num_points:
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = mesh.vertices
                if pcd != None and sampling_method == 'fps':
                    point_cloud = pcd.farthest_point_down_sample(sample_num_points)
                else: point_cloud = mesh.sample_points_poisson_disk(sample_num_points, pcl=pcd)
        else:
            point_cloud.points = mesh.vertices

    else:
        point_cloud = o3d.io.read_point_cloud(file_name)
        print(f"printed point cloud: {point_cloud}")
        if sample_points:
            point_cloud = point_cloud.farthest_point_down_sample(sample_num_points)

    return np.asarray(point_cloud.points)

def k_nearest_neighbors(point_cloud: np.array, 
                        center_point: np.array, 
                        k: int,) -> (np.array, list[int]):
    """
    Description
    -----------
    Finds k nearest neighbors in order (including the center point).

    All points should be 3D points.

    Parameters
    ----------
    point_cloud: np.array
        Point cloud, from which the nearest neighbors will be selected.
        This is in n x 3 numpy array.
    center_point: np.array
        The reference point, from which k nearest neighbors will be found.
        This is in 1 x 3 numpy array.
    k: int
        Number of nearest neighbors (including the reference point).

    Returns
    -------
    (np.array, list[int])
        k nearest neighbors (including the center point) in a k x 3 numpy array.
        List of indices (in reference to the original point cloud) of the k nearest neighbors in order.
    """

    distances = np.linalg.norm(point_cloud - np.broadcast_to(center_point, shape=point_cloud.shape), axis = 1)
    num_points = len(point_cloud)
    k_nearest_neighbors_indices = []

    for _ in range(k):
        min_distance = np.inf
        index = 0
        for point in range(num_points):
            if distances[point] <= min_distance:
                min_distance = distances[point]
                index = point

        distances[index] = np.inf
        k_nearest_neighbors_indices.append(index)
        
    return point_cloud[k_nearest_neighbors_indices, :], k_nearest_neighbors_indices

def patch_sorted_gram_matrix(points: np.array, selected_point: Optional[np.array]=None) -> list[Optional[np.array]]:
    """
    Description
    -----------
    Calculates Sorted Gram Matrix of points, which are in n x 3 numpy array format.

    The calculated Sorted Gram Matrix is that presented in the SGMNet paper. 
    This is per patch SGM, for which the positions of the points in the patch are
    normalized by subtracting the center point position.

    If selected_point is not None, create a patch using the method for points 
    by broadcasting the selected_point to the same shape as points.

    Parameters
    ----------
    points: np.array
        3D points in a n x 3 numpy array format.
        The first row is the center point of the patch.
    selected_point: Optional[np.array]
        3D point in a 1 x 3 numpy array format.
        
    Returns
    -------
    list[Optional[np.array]]
        Sorted Gram Matrix in n x n numpy format.
        If selected_point is not None, also sorted gram matrix using the 
        broadcasted version of this point in the same shape.
    """
    selected_point_sgm = None
    
    center_normalized_points = points - np.broadcast_to(points[0, :], shape=points.shape)
    gm = center_normalized_points @ np.transpose(center_normalized_points)
    sgm = np.sort(gm, axis=1)

    if selected_point is not None:
        selected_points = np.broadcast_to(selected_point, points.shape)
        selected_point_sgm = selected_points @ np.transpose(selected_points)

    return sgm, selected_point_sgm



def patchify_point_cloud(point_cloud: np.array, k: int=30, num_patches: Optional[int]=None) -> (np.array, list[int], list[list[int]]):
    """
    Description
    -----------
    Get patchified point cloud along with the patch centers from original point cloud.
    Default number of patches is (2 * (original number of points // k)). This calculation follows Point-MAE paper.

    Patch centers are sampled using Farthest Point Sampling (FPS), which is a greedy algorithm.
    Go through each center point in FPS and find k nearest neighbors in sorted order.
    Patchify k nearest neighbors (including the center point) using Sorted Gram Matrix.
    Keep track of these patches using a numpy array.
    Keep track of each center point coordinates in a separate numpy array.

    Parameters
    ----------
    point_cloud: np.array
        Original input point cloud that will be patchified in N x 3 numpy array format.
    k: int
        Number of points per patch (Including the center point).
    num_patches: Optional[int]
        Number of patches (ie. Number of kNN clusters).

    Returns
    -------
    (np.array, list[int], list[list[int]])
        Let number of patches be n.
        n x k x k numpy array, where each patch is a k x k Sorted Gram Matrix.
        List of int, where each element represents the index of a center point in the original point cloud.
        List of list of indices (in reference to the original point cloud) of the k nearest neighbors in order
        for each center point.
    """
    points = np.copy(point_cloud)
    if num_patches is None:
        num_patches = 2 * -(len(points)//-k)

    patchified_point_cloud = np.empty((num_patches, k, k))
    patchified_point_cloud_centers = np.empty((num_patches, k, k))
    patch_center_indices = []
    per_patch_indices = []

    num_points = len(points)
    distances = [np.inf for i in range(num_points)]
    
    farthest_index = 0
    for patch in range(num_patches):
        patch_center_indices.append(farthest_index)
        selected_point = points[farthest_index]
        knns, knns_points_indices = k_nearest_neighbors(point_cloud=points, 
                                                        center_point=selected_point, 
                                                        k=k)
        per_patch_indices.append(knns_points_indices)
        sgm, selected_point_sgm = patch_sorted_gram_matrix(knns, selected_point)
        patchified_point_cloud[patch] = sgm
        patchified_point_cloud_centers[patch] = selected_point_sgm

        max_dist = 0
        for point in range(num_points):
            dist = np.linalg.norm(points[point] - selected_point)
            distances[point] = np.min((distances[point], dist))

            if distances[point] > max_dist:
                max_dist = distances[point]
                farthest_index = point

    return patchified_point_cloud, \
           patchified_point_cloud_centers, \
           np.array(patch_center_indices).astype(np.int32), \
           np.array(per_patch_indices).astype(np.int32)

def normalize_positions_with_centroid(point_cloud: np.array) -> np.array:
    """
    Description
    -----------
    Normalizes the point cloud by subtracing the centroid of the point cloud from every point.

    Parameters
    ----------
    point_cloud: np.array
        Point cloud in a N x 3 numpy array format.

    Returns
    -------
    np.array
        Returns a normalized point cloud in a N x 3 numpy array format.
    """
    centroid = np.mean(point_cloud, axis=0)
    return point_cloud - np.broadcast_to(centroid, shape=point_cloud.shape)

def normalize_scale_to_sphere(point_cloud: np.array, radius: np.float32=1.0) -> np.array:
    """
    Description
    -----------
    The given point cloud should have already been centered to the origin.
    Normalize the scale of the points of the given point cloud by dividing each point by the maximum distance from the origin to the points.

    Parameters
    ----------
    point_cloud: np.array
        The given N x 3 point cloud, which should already be centered at the origin.
    radius: np.float32
        Radius of the sphere, in which the normalized point cloud will reside. 

    Returns
    -------
    np.array
        Returns scale-normalized N x 3 point cloud.  
    """
    max_point_dist = np.linalg.norm(np.max(point_cloud, axis=0)) + 1e-8
    return point_cloud / max_point_dist

def get_eigenvectors(point_cloud: np.array, num_eigenvectors: int=500, n_neighbors: int=30) -> np.array:
    """
    Description
    -----------
    Calculates eigenvectors of the Laplacian matrix of the N x 3 numpy array format point cloud.
    The calculated eigenvectors are associated with the smallest eigenvalues in order (from the smallest).

    Parameters
    ----------
    point_cloud: np.array
        Point cloud in a N x 3 numpy array format.
    num_eigenvectors: int
        Number of eigenvectors to calculate.
    n_neighbors: int
        Number of neighbors to consider when calculating Laplacian matrix of the point cloud.

    Returns
    -------
    np.array
        Returns a matrix of eigenvectors as columns in a N x num_eigenvectors numpy array format.
    """
    L, _ = robust_laplacian.point_cloud_laplacian(point_cloud, mollify_factor=1e-5, n_neighbors=n_neighbors)
    _, eigenVecs = sla.eigsh(L, num_eigenvectors, sigma=1e-8, which='LM')
    return eigenVecs

def get_eigenspace_point_cloud(eigenvecs: np.array, point_cloud: np.array) -> np.array:
    """
    Description
    -----------
    Projects the point cloud to eigenspace of the Laplacian matrix.

    Parameters
    ----------
    eigenvecs: np.array
        Eigenvectors of the point cloud in N x num_eigenvectors numpy array format.
        Eigenvectors are columns of the given array, ordered with respect to their eigenvalues
        (from the smallest).
    point_cloud: np.array
        Point cloud in a N x 3 numpy array format.

    Returns
    -------
    np.array
        Returns a eigenspace-projected point cloud in a num_eigenvectors x 3 numpy array format.
    """
    return np.transpose(eigenvecs) @ point_cloud

def eigenspace_point_cloud_reprojection(eigenvecs: np.array, eigen_pcl: np.array) -> np.array:
    """
    Description
    -----------
    Reprojects the point cloud from eigenspace of the Laplacian matrix to original Euclidean space.

    Parameters
    ----------
    eigenvecs: np.array
        Eigenvectors of the point cloud in N x num_eigenvectors numpy array format.
        Eigenvectors are columns of the given array, ordered with respect to their eigenvalues
        (from the smallest).
    eigen_pcl: np.array
        Point cloud in a num_eigenvectors x 3 numpy array format.
        ie. Point cloud projected onto eigenspace.

    Returns
    -------
    np.array
        Returns a reprojected point cloud in a N x 3 numpy array format.
    """

    return eigenvecs @ eigen_pcl
