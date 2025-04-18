import numpy as np
from scipy.spatial.transform import Rotation as R

def _np_random_rotate(point_cloud: np.array, verbose: bool=False) -> np.array:
    """
    Description
    -----------
    Randomly rotate the given point cloud.

    Parameters
    ----------
    point_cloud: np.array
        Point cloud in N x 3 numpy array format.
    verbose: bool
        Whether to display message

    Returns
    -------
    np.array
        Returns a randomly rotated point cloud in N x 3 numpy array format.
    """
    if verbose: print("Random Rotate Applied!\n")
    random_R = R.random().as_matrix()
    return np.transpose(random_R @ np.transpose(point_cloud))

def _np_random_translation(point_cloud: np.array, verbose: bool=False) -> np.array:
    """
    Description
    -----------
    Randomly translate the given point cloud.

    Parameters
    ----------
    point_cloud: np.array
        Point cloud in N x 3 numpy array format.
    verbose: bool
        Whether to display message

    Returns
    -------
    np.array
        Returns a randomly translated point cloud in N x 3 numpy array format.
    """
    if verbose: print("Random Translation Applied!\n")
    random_T = np.random.uniform(low=-0.2, high=0.2, size=(1, 3))
    return point_cloud + np.broadcast_to(random_T, shape=point_cloud.shape)

def _np_random_scaling(point_cloud: np.array, verbose: bool=False) -> np.array:
    """
    Description
    -----------
    Randomly scale the given point cloud.

    Parameters
    ----------
    point_cloud: np.array
        Point cloud in N x 3 numpy array format.
    verbose: bool
        Whether to display message

    Returns
    -------
    np.array
        Returns a randomly scaled point cloud in N x 3 numpy array format.
    """
    if verbose: print("Random Scaling Applied!\n")
    random_S = np.random.uniform(low=0.66, high=1.5)
    return point_cloud * random_S

def _np_random_gaussian_noise(point_cloud: np.array, verbose: bool=False) -> np.array:
    """
    Description
    -----------
    Randomly perturbs the given point cloud with Gaussian noise.

    Parameters
    ----------
    point_cloud: np.array
        Point cloud in N x 3 numpy array format.
    verbose: bool
        Whether to display message

    Returns
    -------
    np.array
        Returns a randomly perturbed point cloud in N x 3 numpy array format.
    """
    if verbose: print("Gaussian Noise Applied!\n")
    gaussian_noise = np.random.normal(loc=-0.0079, scale=0.0048, size=point_cloud.shape)
    return point_cloud + gaussian_noise

def random_point_cloud_augmentation(point_cloud, verbose: bool=False):
    """
    Description
    -----------
    Randomly applies a sequence of random augmentations to the given point cloud.
    
    Possibly applied augmentations are:\n
        Addition of Gaussian noise\n
        Random scaling\n
        Random rotation\n
        Random translation

    Parameters
    ----------
    point_cloud:
        Point cloud in a N x 3 format.
    verbose: bool
        Whether to display message

    Returns
    -------
    Returns a randomly augmented point cloud in N x 3 format.
    """
    augs = [_np_random_gaussian_noise,
            _np_random_scaling, 
            _np_random_rotate, 
            _np_random_translation,]

    rng = np.random.default_rng()
    apply_aug = rng.integers(2, size=len(augs))

    for i in range(len(augs)):
        if apply_aug[i]:
            point_cloud = augs[i](point_cloud, verbose)

    return point_cloud
