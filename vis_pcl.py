import numpy as np
import polyscope as ps
import robust_laplacian
import open3d as o3d
from data_process.utils.data_augment import *
from data_process.utils.data_process import *
import scipy.sparse.linalg as sla
import matplotlib.pyplot as plt
from typing import Optional



def _view_eigenspace_value_distribution(projected_V: np.array, bins: int=5000, axis: int=1) -> None:
    """
    Description
    -----------
    Given a projected vertices (to eigenspace) and a corresponding axis, 
    plot a histogram of the values of the eigenspace-projected vertices' components
    at the specified axis. 

    Parameters
    ----------
    projected_V: np.array
        k x 3 numpy array format.
        Eigenspace-projected vertices.
    bins: int
        Number of bins to use to plot a histogram
    axis: int
        Axis of the projected vertices. (0 if x, 1 if y, 2 if z).

    Returns
    -------
    None
    """
    _ = plt.hist(projected_V[:, axis], bins='auto')

    hist, bin_edges = np.histogram(projected_V[:, axis], bins=bins, density=True)
    bin_centres = (bin_edges[:-1] + bin_edges[1:])/2
    plt.plot(bin_centres, hist, label='Test data')

    plt.show()

    return None

def _view_point_cloud(point_cloud1: np.array, 
                      evecs1: Optional[np.array]=None,
                      point_cloud2: Optional[np.array]=None, 
                      evecs2: Optional[np.array]=None,
                      mesh_exists: bool=False,
                      mesh: Optional[o3d.geometry.TriangleMesh]=None) -> None:
    """
    Description
    -----------
    Given a point cloud with its respective eigenvectors, 
    this function gives visualization of the point cloud.
    Optionally, a second point cloud can be viewed together.

    Parameters
    ----------
    point_cloud1: np.array
        N x 3 numpy array format of point cloud.
    evecs1: Optional[np.array]
        This is for point_cloud1.
        N x k numpy array format.
        Where N is the eigenvector dimension and k is the number of eigenvectors.
    point_cloud2: Optional[np.array]
        N x 3 numpy array format of point cloud.
    evecs2: Optional[np.array]
        This is for point_cloud2.
        N x k numpy array format.
        Where N is the eigenvector dimension and k is the number of eigenvectors.
    mesh_exists: bool
        Indicates that mesh exists or not.
    mesh: Optional[o3d.geometry.TriangleMesh]
        Mesh.
    
    Returns 
    -------
    None
    """
    ps.init()
    
    pcd1 = ps.register_point_cloud("Point Cloud", point_cloud1)
    if evecs1 is not None and evecs1.shape[0] == len(point_cloud1):
        for i in range(evecs1.shape[1]):
            pcd1.add_scalar_quantity("eigenvector_"+str(i), evecs1[:, i], enabled=True)
    
    if point_cloud2 is not None:
        pcd2 = ps.register_point_cloud("Recon Point Cloud", point_cloud2)
        if evecs2 is not None and evecs2.shape[0] == len(point_cloud2):
            for i in range(evecs2.shape[1]):
                pcd2.add_scalar_quantity("eigenvector_"+str(i), evecs2[:, i], enabled=True)
    if mesh_exists and mesh.vertices is not None and mesh.triangles is not None:
        ps.register_surface_mesh('mesh', np.asarray(mesh.vertices), np.asarray(mesh_f), smooth_shade=True)
    
    ps.show()

    return None

def view_sampled_pcl_kNN_centers(file: str, sample_num_points: int=6000, k: int=30) -> None:
    """
    Description
    -----------
    Sample given number of points from the given mesh vertices, as well as
    centers that will be used for kNN. Visualize them independently.

    Parameters
    ----------
    file: str
        File to a point cloud. This should be a full path to the file.
    sample_num_points: int
        Number of points to sample.
    k: int
        k used in kNN.

    Returns
    -------
    None
    """
    mesh = o3d.io.read_triangle_mesh(file)
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices

    point_cloud1 = pcd.farthest_point_down_sample(sample_num_points)
    point_cloud2 = point_cloud1.farthest_point_down_sample(int(2 * (sample_num_points // k)))

    _view_point_cloud(point_cloud1= np.asarray(point_cloud1.points), 
                      point_cloud2= np.asarray(point_cloud2.points),)

def view_point_cloud(file: str, 
                     normalize: bool=True, 
                     random_augment: bool=True, 
                     n_eig: int=500, 
                     view_recon: bool=False,
                     axis: str='x',
                     verbose: bool=True, 
                     sample_num_points: int=6000,
                     mesh_exists: bool=False) -> None:
    """
    Description
    -----------
    Given a file to a point cloud, this function gives visualization of the associated point cloud.
    This can be viewed along with the value distribution of the eigenspace-projected vertices
    at a specified axis.

    Optionally, a reconstructed view using the eigenvectors can be viewed
    alongside the original point cloud.

    Parameters
    ----------
    file: str
        File to a point cloud. This should be a full path to the file.
    normalize: bool
        Normalize the original point cloud.
        This consists of centering the point cloud to the origin and
        scaling the points so that they all fall inside a unit sphere.
    random_augment: bool
        Apply random augmentations.
        This could be any combination of gaussian noise addition, translation, scaling, and rotation.
    n_eig: int
        Number of eigenvectors to use to reconstruct the original point cloud.
    view_recon: bool
        View the reconstructed original point cloud.
    axis: str
        Either 'x', 'y', or 'z'.
        Axis of the eigenspace-projected vertices to view the value distribution of the
        eigenspace-projected vertices at the specified axis.
    verbose: bool
        To display messages regarding the type of random augmentations that were applied.
    sample_num_points: int
        Number of points to sample.
    mesh_exists: bool
        Indicates that mesh exists or not.
    
    Returns 
    -------
    None
    """
    mesh = None
    V = None
    if mesh_exists:
        mesh = o3d.io.read_triangle_mesh(file)
    else:
        V = read_point_cloud(file_name=file, is_mesh=False, sample_num_points=sample_num_points)

    if random_augment:
        V = random_point_cloud_augmentation(V, verbose)
    if normalize:
        V = normalize_scale_to_sphere(V)
    
    if verbose: print("Calculating laplacian matrix...")
    L, _ = robust_laplacian.point_cloud_laplacian(V, mollify_factor=1e-5, n_neighbors=30)

    if verbose: print("Calculating eigenvectors...")
    _, evecs = sla.eigsh(L, n_eig, sigma=1e-8, which='LM')

    if verbose: print("Projecting vertices onto eigenspace...")
    projected_V = np.matmul(np.transpose(evecs), V)

    if verbose: print("Reconstructing vertices...")
    recon_V = None
    if view_recon:
        recon_V = np.matmul(evecs, projected_V)

    axis_dict = {
        'x': 0,
        'y': 1,
        'z': 2
    }
    if verbose: print(f"Value distribution for projected {axis}-axis vertices")
    #_view_eigenspace_value_distribution(projected_V, bins=5000, axis=axis_dict[axis])

    _view_point_cloud(point_cloud1=V, 
                      evecs1=evecs, 
                      point_cloud2=recon_V, 
                      mesh_exists=mesh_exists, 
                      mesh=mesh)

    return None

if __name__ == "__main__":
    
    file = os.path.join(os.getcwd(), 'sample_data/mast3r_foot_raw_data_gwang.ply') # 5 images
    #file = os.path.join(os.getcwd(), 'sample_data/mast3r_foot_raw_data_yang.ply') # 5 images
    #file = os.path.join(os.getcwd(), 'sample_data/mast3r_right_with_floor.ply') # 3 images

    view_point_cloud(file=file, 
                     normalize=False,
                     random_augment=False,
                     n_eig=200, 
                     view_recon=True, 
                     axis='y',
                     verbose=True,
                     sample_num_points=60000,
                     mesh_exists=False)
    """
    file = os.path.join(os.getcwd(), 'sample_data/mast3r_without_floor_gwang_5_images.ply')
    #file = os.path.join(os.getcwd(), 'sample_data/mast3r_without_floor_yang_3_images.ply')
    view_point_cloud(file=file, 
                     normalize=True,
                     random_augment=False,
                     n_eig=5, 
                     view_recon=False, 
                     axis='y',
                     verbose=True,
                     sample_num_points=30000,
                     mesh_exists=False)
    """