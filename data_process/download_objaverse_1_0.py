import objaverse.xl as oxl
import multiprocessing as mp
import trimesh
import os
import shutil
from typing import Any, Dict, Hashable


in_dir = os.listdir('D:\objaverse10')
random_state = 824

def handle_found_object(
    local_path: str,
    file_identifier: str,
    sha256: str,
    metadata: Dict[Hashable, Any]
) -> None:
    """
    Description
    -----------
    Completion handler for oxl.download_objects method's handle_found_object argument.
    Moves the downloaded file to a folder of choice ("folder of choice" should be assigned within this method
    by changing the "download_dir" variable).

    Parameters
    ----------
    local_path: str
        Local path to the downloaded 3D object.
    file_identifier: str
        File identifier of the 3D object.
    sha256: str
        sha256 of the contents of the 3D object.
    metadata: Dict[Hashable, Any]
        Metadata about the 3D object, such as the GitHub organization and repo name.

    Returns
    -------
    None
    """
    try:
        download_dir = 'D:\objaverse10'
        obj_name_split = local_path.split("/")
        file_name = obj_name_split[len(obj_name_split)-1].split('.')[0]
        if file_name in in_dir:
            return None
        dir_name = os.path.join(download_dir, file_name)
        os.mkdir(dir_name)

        mesh = trimesh.load(local_path, force='mesh')
        
        if isinstance(mesh, list):
            mesh = trimesh.util.concatenate(mesh)
            if len(mesh.vertices) == 0:
                shutil.rmtree(dir_name)
                return None
        if len(mesh.vertices) < 2500:
            shutil.rmtree(dir_name)
            return None
        
        mesh.export(os.path.join(dir_name, file_name+'.obj'))
    except:
        download_dir = 'D:\objaverse10'
        obj_name_split = local_path.split("/")
        file_name = obj_name_split[len(obj_name_split)-1].split('.')[0]
        dir_name = os.path.join(download_dir, file_name)
        shutil.rmtree(dir_name)
        return None
    
def download_objaverse_multiprocess(download_dir: str='D:\objaverse', download_num: int=200000, partial_download_num: int=100) -> None:
    """
    Description
    -----------
        Downloads 3D object files from objaverse.
        Files are downloaded from [github, smithsonian, thingiverse, sketchfab].

    Parameters
    ----------
    download_dir: str
        Directory to which the files are downloaded.
    download_num: int
        Total number of files to download.
    partial_download_num: int
        Number of files to download at a time. 

    Returns
    -------
    None
    """
    num_processes = max(1, mp.cpu_count() - 6)
    
    if mp.cpu_count() < partial_download_num:
        partial_download_num = max(1, num_processes - 3)
    annotations = oxl.get_annotations(download_dir=download_dir) # default download directory
    github_indices = annotations[annotations['source'].isin(['github'])].index
    annotations.drop(github_indices, inplace=True)
    print(f"Annotation length: {len(annotations)}")
    for i in range(download_num // partial_download_num):
        sampled_df = annotations.sample(n=100, random_state=random_state).reset_index(drop=True)
        sampled_df_indices = annotations[annotations['fileIdentifier'].isin(sampled_df['fileIdentifier'])].index
        annotations.drop(sampled_df_indices, inplace=True)
        oxl.download_objects(objects=sampled_df, download_dir=download_dir, processes=num_processes, handle_found_object=handle_found_object, save_repo_format='files')
        print(f"Annotation length: {len(annotations)}")

if __name__ == "__main__":
    download_objaverse_multiprocess()
    