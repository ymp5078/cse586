import shutil
import os
from tqdm import tqdm
import sys

def clean_dir(map_dir,prefix='ResNet50_E_ImageNet_'):
    all_files = [f for f in os.listdir(map_dir) if f!='base_map']
    all_files = sorted(all_files,key = lambda x: int(x.split('_')[-1]))
    all_files = all_files[:-2]
    # print(all_files)
    for f in tqdm(all_files):
        shutil.rmtree(os.path.join(map_dir,f))


if __name__=='__main__':
    print(sys.argv)
    if len(sys.argv)>1:
        clean_dir(sys.argv[1])