import multiprocessing as mp
from multiprocessing import Process
import os
import numpy as np
from utils import data_process, data_augment
import random
import shutil

def remove_file(dir: str, file: str) -> None:
    file = os.path.join(dir, file)
    if os.path.exists(file):
        os.remove(file)
    return None

def remove_files(point_cloud_dir: str) -> None:
    remove_file(point_cloud_dir, 'unmasked_input.npy')
    remove_file(point_cloud_dir, 'unmasked_points.npy')
    remove_file(point_cloud_dir, 'masked_points.npy')
    remove_file(point_cloud_dir, 'input_positions.npy')
    remove_file(point_cloud_dir, 'vae_reproj_tensor.npy')
    remove_file(point_cloud_dir, 'vae_target_z.npy')
    remove_file(point_cloud_dir, 'vae_target_recon.npy')
    remove_file(point_cloud_dir, 'unmasked_token_indices.npy')
    remove_file(point_cloud_dir, 'masked_token_indices.npy')
    return None

def vae_data(point_cloud: np.array, point_cloud_dir: str) -> None:
    points = np.copy(point_cloud)
    num_vae_points = len(points) // 2
    rng = np.random.default_rng()
    if rng.integers(2) == 0:
        # Randomly sample half the original points.
        points = np.take(a=points,
                        indices=random.sample(list(range(len(points))), 
                                              num_vae_points),
                        axis=0)
    else:
        # Sample half the original points using FPS sampling algorithm
        points = data_process.fps_sample_numpy_point_cloud(point_cloud=points, 
                                                           sample_num_points=num_vae_points)
    
    unmasked_input, input_positions, _, _ = \
        data_process.patchify_point_cloud(point_cloud=points)
    
    eigenvectors = data_process.get_eigenvectors(points)
    eigenspace_points = data_process.get_eigenspace_point_cloud(eigenvecs=eigenvectors, 
                                                                point_cloud=points)
    reprojected_points = data_process.eigenspace_point_cloud_reprojection(eigenvecs=eigenvectors, 
                                                                          eigen_pcl=eigenspace_points)
    vae_dir = os.path.join(point_cloud_dir, 'vae')
    os.makedirs(vae_dir, exist_ok=True)
    np.save(os.path.join(vae_dir, 'unmasked_input'), unmasked_input.astype(np.float32))
    np.save(os.path.join(vae_dir, 'input_positions'), input_positions.astype(np.float32))
    np.save(os.path.join(vae_dir, 'vae_reproj_tensor'), eigenvectors.astype(np.float32))
    np.save(os.path.join(vae_dir, 'vae_target_z'), eigenspace_points.astype(np.float32))
    np.save(os.path.join(vae_dir, 'vae_target_recon'), reprojected_points.astype(np.float32))
    return None

def mae_data(point_cloud: np.array, point_cloud_dir: str) -> None:
    points = np.copy(point_cloud)
    points_sgm, input_positions, patch_center_indices, per_patch_indices = \
        data_process.patchify_point_cloud(point_cloud=points)
    
    len_patch_center_indices = len(patch_center_indices)
    token_indices = list(range(len(patch_center_indices)))

    # Input to Deep Learning model
    unmasked_token_indices = random.sample(token_indices, len_patch_center_indices // 2)
    masked_token_indices = list(set(token_indices) - set(unmasked_token_indices))

    # Intermediate values
    unmasked_point_indices = np.take(a=per_patch_indices,
                                     indices=unmasked_token_indices,
                                     axis=0).flatten().tolist()
    masked_point_indices = np.take(a=per_patch_indices,
                                     indices=masked_token_indices,
                                     axis=0).flatten().tolist()

    # Input to deep learning model
    unmasked_input = np.take(a=points_sgm,
                             indices=unmasked_token_indices,
                             axis=0)
    
    # Used as deep learning model target
    unmasked_points = np.take(a=points,
                              indices=unmasked_point_indices,
                              axis=0)
    masked_points = np.take(a=points,
                            indices=masked_point_indices,
                            axis=0)
    
    mae_dir = os.path.join(point_cloud_dir, 'mae')
    os.makedirs(mae_dir, exist_ok=True)
    np.save(os.path.join(mae_dir, 'unmasked_input'), unmasked_input.astype(np.float32))
    np.save(os.path.join(mae_dir, 'input_positions'), input_positions.astype(np.float32))
    np.save(os.path.join(mae_dir, 'unmasked_points'), unmasked_points.astype(np.float32))
    np.save(os.path.join(mae_dir, 'masked_points'), masked_points.astype(np.float32))
    
    np.save(os.path.join(mae_dir, 'unmasked_token_indices'), 
            np.array(unmasked_token_indices).astype(np.int32))
    np.save(os.path.join(mae_dir, 'masked_token_indices'), 
            np.array(masked_token_indices).astype(np.int32))
    return None

def seg_data(points: np.array, point_cloud_dir: str) -> None:
    return None
    
def process_obj_list(data_dir: str, obj_list: list[str], process_id: int=1):
    """
    """
    length = len(obj_list)
    for i, obj_name in enumerate(obj_list):
        point_cloud_dir = os.path.join(data_dir, obj_name)
        file_name = os.path.join(point_cloud_dir, obj_name + '.obj')
        
        point_cloud = None
        try:
            point_cloud = data_process.read_point_cloud(file_name=file_name,
                                                        sample_num_points=12000)
        except:
            shutil.rmtree(point_cloud_dir)
            continue

        point_cloud = data_process.normalize_positions_with_centroid(point_cloud)
        point_cloud = data_process.normalize_scale_to_sphere(point_cloud)

        not_succeeded = True
        failed = 0
        while not_succeeded:
            try:
                vae_data(point_cloud, point_cloud_dir)
                if (i + 1) % 100 == 0:
                    print(f"process_id {process_id} succeeded {i + 1} / {length} with {failed} last failures")
                not_succeeded = False
            except:
                failed += 1
                if failed == 1:
                    print(f"process_id {process_id} still at {i + 1} / {length} failed {failed} times")
                continue
    return None

def vae_data_prep(data_dir: str, num_workers: int=1,) -> None:
    """
    Description
    -----------

    Parameters
    ----------
    data_dir: str

    num_workers: int
    
    
    Returns
    -------
    None
    """
    print(num_workers)
    obj_list = os.listdir(data_dir)
    if num_workers == 1:
        process_obj_list(data_dir, obj_list)
    else:
        num_processes = max(1, num_workers)
        print(f'Number of processes: {num_processes}')
        processes = []
        for i in range(num_processes):
            sub_obj_list = [obj_name for j, obj_name in enumerate(obj_list) if j % num_processes == i]
            p = Process(target=process_obj_list, args=(data_dir, sub_obj_list, i,))
            processes.append(p)
            processes[i].start()
        
        for i in range(num_processes):
             processes[i].join()
    return None
    

if __name__ == '__main__':
    vae_data_prep(data_dir='D:/teeth_segmentation/data/data/train', num_workers=12)