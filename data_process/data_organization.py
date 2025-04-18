import os
import shutil
import fnmatch
import random
import trimesh


def organize_data(src: str, dest: str, train_p: float=0.9) -> None:
    objs = []
    i = 40317

    print("Walking through source directory...")
    for dirpath, _, files in os.walk(src):
        for file in fnmatch.filter(files, '*.obj'):
            objs.append([os.path.join(dirpath, file), f'shapenet_core_{i}'])
            i += 1
    
    random.shuffle(objs)
    objs_len = len(objs)
    train_proportion = round(objs_len * train_p)
    train_objs_path = os.path.join(dest, 'train')
    test_objs_path = os.path.join(dest, 'test')
    train_objs = []
    test_objs = []
    
    print("Moving files to destination...")
    for i, obj in enumerate(objs):
        obj_file = obj[1] + '.obj'
        if i < train_proportion:
            obj_dir = os.path.join(train_objs_path, obj[1])
            os.mkdir(obj_dir)
            shutil.copyfile(obj[0], os.path.join(obj_dir, obj_file))
            train_objs.append(obj_dir)
        else: 
            obj_dir = os.path.join(test_objs_path, obj[1])
            os.mkdir(obj_dir)
            shutil.copyfile(obj[0], os.path.join(obj_dir, obj_file))
            test_objs.append(obj_dir)
    
    return None

if __name__ == "__main__":
    #organize_data(src=r'D:\temp', dest=r'D:\teeth_segmentation/pre_train/vae')