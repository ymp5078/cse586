from PIL import Image
import os
import os.path
import numpy as np
import pickle
import torch
from typing import Any, Callable, Optional, Tuple, Dict, List, Iterator
import codecs
from pathlib import Path
import json
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.folder import ImageFolder
from torchvision.datasets.utils import check_integrity, download_and_extract_archive,verify_str_arg, extract_archive
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms
import sys
import inspect
import tempfile
from contextlib import contextmanager
import shutil
from joblib import Parallel, delayed
from scipy import ndimage

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
# print(sys.path)
from data.utils import attribute_image_features,attribute_image_object


def read_label_file(path: str) -> torch.Tensor:
    x = read_sn3_pascalvincent_tensor(path, strict=False)
    # assert(x.dtype == torch.uint8)
    # assert(x.ndimension() == 1)
    return x.astype(np.int64)


def read_image_file(path: str) -> torch.Tensor:
    x = read_sn3_pascalvincent_tensor(path, strict=False)
    # assert(x.dtype == torch.uint8)
    # assert(x.ndimension() == 3)
    return x

def read_sn3_pascalvincent_tensor(path: str, strict: bool = True) -> torch.Tensor:
    """Read a SN3 file in "Pascal Vincent" format (Lush file 'libidx/idx-io.lsh').
       Argument may be a filename, compressed filename, or file object.
    """
    # read
    with open(path, "rb") as f:
        data = f.read()
    # parse
    magic = get_int(data[0:4])
    nd = magic % 256
    ty = magic // 256
    assert 1 <= nd <= 3
    assert 8 <= ty <= 14
    m = SN3_PASCALVINCENT_TYPEMAP[ty]
    s = [get_int(data[4 * (i + 1): 4 * (i + 2)]) for i in range(nd)]
    parsed = np.frombuffer(data, dtype=m[1], offset=(4 * (nd + 1)))
    assert parsed.shape[0] == np.prod(s) or not strict
    return np.reshape(parsed.astype(m[2]),s)

def get_int(b: bytes) -> int:
    return int(codecs.encode(b, 'hex'), 16)

SN3_PASCALVINCENT_TYPEMAP = {
    8: (torch.uint8, np.uint8, np.uint8),
    9: (torch.int8, np.int8, np.int8),
    11: (torch.int16, np.dtype('>i2'), 'i2'),
    12: (torch.int32, np.dtype('>i4'), 'i4'),
    13: (torch.float32, np.dtype('>f4'), 'f4'),
    14: (torch.float64, np.dtype('>f8'), 'f8')
}

def erase_by_map(img: Any,saliency_map: Any, smooth_map: bool,map_out: bool, kernel_size: int = 5, sigma: float = 5.0,noisey: bool = False, channel_last: bool = True) -> Any:
    """
        img: (H,W,C)
        saliency_map: (H,W)
        smooth_map: bool
        map_out: bool
    """
    if len(img.shape)==3:
        C=3
    else:
        C=1
        img = np.expand_dims(img,-1)
    # print(img.shape)
    if channel_last:
        c_mean = np.expand_dims(np.mean(img.reshape((-1,C)),axis=0),[0,1])
    else:
        c_mean = np.expand_dims(np.mean(img.reshape((C,-1)),axis=1),[1,2])
    # print(img.shape)
    if noisey:
        c_mean = np.random.uniform(low=img.min(), high=img.max(),size=img.shape).astype(np.float32)*c_mean
        # print('cmean shape',np.expand_dims(c_mean,0).transpose([0,3,1,2]).shape)
        # c_mean = transforms.functional.gaussian_blur(torch.from_numpy(np.expand_dims(c_mean,0).transpose([0,3,1,2])),kernel_size=kernel_size,sigma=sigma).numpy().transpose([0,2,3,1])
    saliency_map = torch.from_numpy(np.expand_dims(saliency_map, 0).astype(np.float32))
    if img.shape[1] != saliency_map.shape[1]:
        # print(saliency_map.shape,img.shape[:2])
        saliency_map = torch.nn.functional.conv_transpose2d(torch.unsqueeze(saliency_map,0),weight=torch.ones(1,1,9,9),stride = 9,padding=1,output_padding=1)
        # saliency_map = saliency_map[:224,:224]
        # saliency_map = torch.nn.functional.pad(saliency_map,pad=(4,4,4,4),value=0)
        # print(saliency_map.sum())
    if smooth_map:
        if noisey:
            # print(saliency_map.shape,c_mean.shape)
            if c_mean.shape[0] == 3:
                c_mean[:,(1-np.squeeze(saliency_map))>0.5]=0.0
                c_mean = transforms.functional.gaussian_blur(torch.from_numpy(np.expand_dims(c_mean,0)),kernel_size=kernel_size,sigma=sigma).numpy()
            else:
                c_mean[(1-np.squeeze(saliency_map))>0.5,:]=0.0
                c_mean = transforms.functional.gaussian_blur(torch.from_numpy(np.expand_dims(c_mean,0).transpose([0,3,1,2])),kernel_size=kernel_size,sigma=sigma).numpy().transpose([0,2,3,1])
    
        saliency_map = transforms.functional.gaussian_blur(saliency_map,kernel_size=kernel_size,sigma=sigma).numpy()
    # print((saliency_map[saliency_map>0]))
    saliency_map = np.squeeze(saliency_map)
    if channel_last:
        saliency_map = np.expand_dims(saliency_map, 2)
        saliency_map = np.tile(saliency_map,[1,1,C])
    else:
        saliency_map = np.expand_dims(saliency_map, 0)
        saliency_map = np.tile(saliency_map,[C,1,1])
    # print(saliency_map.shape)
    if map_out:
        fill_values = saliency_map*c_mean
        img = img*(1-saliency_map)+fill_values # mask out based on the saliency map
        # print(fill_values.shape)
    else:
        fill_values = (1-saliency_map)*c_mean
        img = img*saliency_map + fill_values
    
    return np.squeeze(img)


class E_CIFAR10(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
    ]
    val_list = [
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        map_files: list = None,
        map_dir: str = None,
        map_out: bool = True,
        iter_map: bool = True, # ignored if train
        smooth_map: bool = True,
        kernel_size: int = 5,
        sigma: float = 5.0,
        bootstrap_maps: bool = False,
        use_latest_map: bool = True,
        use_basemap:bool = True,
        cur_iter: int = None,
        pertub: str = 'cmean',
        download: bool = False,
    ) -> None:

        super(E_CIFAR10, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.split = split  # training set or test set
        self.map_out = map_out
        self.smooth_map = smooth_map
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.pertub = pertub
        self.bootstrap_maps = bootstrap_maps
        self.use_latest_map = use_latest_map
        self.use_basemap = use_basemap
        self.cur_iter = cur_iter
        self.iter_map = iter_map if split in ['test','val'] else True
        if map_dir is not None:
            self.map_dir = os.path.join(map_dir,self.split)

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.split == 'train':
            downloaded_list = self.train_list
        elif self.split == 'val':
            downloaded_list = self.val_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.file_names = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                self.file_names.append(file_path)
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])
        self.base_map = None
        if not iter_map:
            print(os.path.join(self.map_dir,'base_map.npy'))
            if os.path.exists(os.path.join(self.map_dir,'base_map.npy')):
                self.base_map = np.load(os.path.join(self.map_dir,'base_map.npy'))
        self.saliency_map = None
        if map_files is not None and map_files!=[]:
            print('Found saliency_map')
            
            # model_dataset = '_'.join(map_files[0].split('_')[:-1])
            # cur_map = len(map_files)-1
            # print(os.path.join(self.map_dir,f'{model_dataset}_{cur_map}.npy'))
            # self.saliency_map_latest = np.load(os.path.join(self.map_dir,f'{model_dataset}_{cur_map}.npy'))# each map is binary
            
            # self.saliency_map = [np.load(os.path.join(self.map_dir,f_path)) for f_path in map_files] # each map is binary
            # self.saliency_map = np.stack(self.saliency_map).max(axis=0)

            if self.cur_iter is not None:
                cur_map = self.cur_iter-1
            else:
                cur_map = len(map_files)-1
            model_dataset = '_'.join(map_files[0].split('_')[:-1])
            # print(os.path.join(self.map_dir,f'{model_dataset}_{cur_map}.npy'))
            self.saliency_map_latest = np.load(os.path.join(self.map_dir,f'{model_dataset}_{cur_map}.npy'))# each map is binary
            
            # self.saliency_map = [np.load(os.path.join(self.map_dir,f_path)) for f_path in map_files] # each map is binary
            self.saliency_map = self.saliency_map_latest

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self._load_meta()

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if self.use_basemap:
            if self.base_map is not None:
                saliency_map = np.expand_dims(1/(1 + np.exp(-self.base_map[index])),-1)
                img = img*(1-saliency_map)
                img = ((img-img.min())/(img.max()-img.min())*255)
            img = Image.fromarray(img.astype(np.uint8))
        elif self.saliency_map is not None:
            if self.use_latest_map:
                saliency_map = self.saliency_map_latest[index]
            else:
                saliency_map = self.saliency_map[index]
            if self.bootstrap_maps:
                num_boots = 10
                img = [Image.fromarray(erase_by_map(img,saliency_map,self.smooth_map,self.map_out,kernel_size = self.kernel_size,sigma=self.sigma,noisey=True).astype(np.uint8)) for _ in range(num_boots)]
            else:
                img = erase_by_map(img,saliency_map,self.smooth_map,self.map_out,kernel_size = self.kernel_size,sigma=self.sigma,noisey = self.pertub!='cmean')
                img = Image.fromarray(img.astype(np.uint8))
        else:
            img = Image.fromarray(img.astype(np.uint8))

        if self.transform is not None:
            if self.bootstrap_maps and self.saliency_map is not None:
                img = [self.transform(i) for i in img]
            else:
                img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target


    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self) -> str:
        return "Split: {}".format(self.split)

    def generate_discrete_saliency_map_parallel(self,model,map_alg,map_path,threshold,drop_max,use_cuda):
        
        def gen_one_map(i,drop_max):
            # if os.path.exists(map_file):
            #     continue
                    # print('found_basemap')
            grayscale_cam = self.base_map[i]
            grayscale_cam =  np.expand_dims(grayscale_cam,0)
            if not drop_max:
                grayscale_cam = -grayscale_cam
            grayscale_cam = grayscale_cam - np.min(grayscale_cam)
            if self.saliency_map is not None:
                grayscale_cam = grayscale_cam * np.expand_dims(1-self.saliency_map[i],0)
            idx = np.dstack(np.unravel_index(np.argsort(grayscale_cam.ravel()), (32, 32)))
            
            ridx,cidx = np.split(idx[0,-int(1024*threshold):,:],2,axis=-1)
            
            out_cam = np.zeros_like(grayscale_cam)
            out_cam[0][ridx,cidx] = 1
            if self.saliency_map is not None:
                out_cam = out_cam + self.saliency_map[i]
            
            return out_cam
        grayscale_cams = Parallel(n_jobs=24,backend='threading')(delayed(gen_one_map)(i,drop_max) for i in range(len(self.base_map)))
        grayscale_cams = np.concatenate(grayscale_cams,axis=0).astype(dtype=np.float32)
        
        np.save(f"{map_path}.npy",grayscale_cams)
        return

    def remove_maps(self,map_path):
        if os.path.exists(f"{map_path}.npy"):
            os.remove(f"{map_path}.npy")
            print(f'deleted {map_path}.npy')

    def generate_discrete_saliency_map(self,model,map_alg,map_path,threshold,drop_max,use_cuda):
        if not self.iter_map and self.base_map is not None:
            return self.generate_discrete_saliency_map_parallel(model,map_alg,map_path,threshold,drop_max,use_cuda)
        try:
            target_layers = model.get_target_layers()
        except:
            target_layers = [model.features[-1]]
        if self.base_map is None:
            map_alg = attribute_image_object(algorithm=map_alg,alg_args={'model':model,'layer':target_layers[0],'multiply_by_inputs':True})
        grayscale_cams = []
        baselines = torch.randn(20, 3, 32, 32)
        base_maps=[]
        if use_cuda:
            baselines = baselines.cuda()
        for i in tqdm(range(self.data.shape[0])):
            input_tensor, target_category = self.__getitem__(i)
            input_tensor = torch.unsqueeze(input_tensor, 0)
            if use_cuda:
                input_tensor = input_tensor.cuda()
            if self.iter_map:
                # print('iter map')
                grayscale_cam = attribute_image_features(attr_obj=map_alg,channel_aggregation='all',
                                                        att_args={'inputs':input_tensor,
                                                                'baselines':baselines,
                                                                'target':target_category})
            else:
                # print('no iter map')
                if self.base_map is None:
                    grayscale_cam = attribute_image_features(attr_obj=map_alg,channel_aggregation='all',
                                                            att_args={'inputs':input_tensor,
                                                                    'baselines':baselines,
                                                                    'target':target_category})
                    base_map = np.expand_dims(grayscale_cam,0)
                    base_maps.append(base_map)
                else:
                    # print('found_basemap')
                    grayscale_cam = self.base_map[i]
            
            grayscale_cam =  np.expand_dims(grayscale_cam,0)
            if not drop_max:
                grayscale_cam = -grayscale_cam
            grayscale_cam = grayscale_cam - np.min(grayscale_cam)
            if self.saliency_map is not None:
                grayscale_cam = grayscale_cam * np.expand_dims(1-self.saliency_map[i],0)
            idx = np.dstack(np.unravel_index(np.argsort(grayscale_cam.ravel()), (32, 32)))
            
            ridx,cidx = np.split(idx[0,-int(1024*threshold):,:],2,axis=-1)
            
            out_cam = np.zeros_like(grayscale_cam)
            out_cam[0][ridx,cidx] = 1
            if self.saliency_map is not None:
                out_cam = out_cam + self.saliency_map[i]
            
            grayscale_cams.append(out_cam)
        grayscale_cams = np.concatenate(grayscale_cams,axis=0)
        if len(base_maps)!=0:
            base_map_dir = os.path.join(self.map_dir,'base_map.npy')
            print(f'save to {base_map_dir}')
            base_maps = np.concatenate(base_maps,axis=0)
            np.save(os.path.join(self.map_dir,'base_map.npy'),base_maps)
        np.save(f"{map_path}.npy",grayscale_cams)

    def generate_discrete_random_saliency_map(self,map_path,threshold):
        dummy_imgs = np.random.rand(self.data.shape[0],32,32)
        if self.saliency_map is not None:
            dummy_imgs = dummy_imgs * (1-self.saliency_map)
        grayscale_cams = []
        # for grayscale_cam in tqdm(dummy_imgs):
        def gen_one_map(i):
            grayscale_cam = dummy_imgs[i]
            idx = np.dstack(np.unravel_index(np.argsort(grayscale_cam.ravel()), (32, 32)))
            ridx,cidx = np.split(idx[0,-int(1024*threshold):,:],2,axis=-1)
            out_cam = np.zeros_like(grayscale_cam)
            out_cam[ridx,cidx] = 1
            if self.saliency_map is not None:
                out_cam = out_cam+self.saliency_map[i]

            # print(idx.shape,grayscale_cam.max())
            return out_cam
            # grayscale_cams.append(out_cam)
        grayscale_cams = Parallel(n_jobs=24,backend='threading')(delayed(gen_one_map)(i) for i in range(len(dummy_imgs)))
        grayscale_cams = np.stack(grayscale_cams,axis=0).astype(dtype=np.float32)
        
        np.save(f"{map_path}.npy",grayscale_cams)


class E_MNIST(VisionDataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``MNIST/processed/training.pt``
            and  ``MNIST/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    mirrors = [
        'http://yann.lecun.com/exdb/mnist/',
        'https://ossci-datasets.s3.amazonaws.com/mnist/',
    ]

    resources = [
        ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
        ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
        ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c")
    ]

    training_file = 'training.pt'
    test_file = 'test.pt'
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        map_files: list = None,
        map_dir: str = None,
        map_out: bool = True,
        iter_map: bool = True, # ignored if train
        smooth_map: bool = True,
        kernel_size: int = 5,
        sigma: float = 5.0,
        bootstrap_maps: bool = False,
        use_latest_map: bool = False,
        use_basemap:bool = False,
        cur_iter: int = None,
        pertub: str = 'cmean',
        download: bool = False,
    ) -> None:
        super(E_MNIST, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        # self.train = train  # training set or test set
        self.split = split  # training set or test set
        self.train = True if self.split in ['train','val'] else False
        self.map_out = map_out
        self.smooth_map = smooth_map
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.pertub = pertub
        self.bootstrap_maps = bootstrap_maps
        self.use_latest_map = use_latest_map
        self.use_basemap = use_basemap
        self.cur_iter = cur_iter
        self.iter_map = iter_map if split in ['test','val'] else True
        if self._check_legacy_exist():
            self.data, self.targets = self._load_legacy_data()
            return

        if map_dir is not None:
            self.map_dir = os.path.join(map_dir,self.split)
        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        self.data, self.targets = self._load_data()
        self.base_map = None
        if not iter_map:
            print(os.path.join(self.map_dir,'base_map.npy'))
            if os.path.exists(os.path.join(self.map_dir,'base_map.npy')):
                self.base_map = np.load(os.path.join(self.map_dir,'base_map.npy'))
        self.saliency_map = None
        if map_files is not None and map_files!=[]:
            print('Found saliency_map')
            if self.cur_iter is not None:
                cur_map = self.cur_iter-1
            else:
                cur_map = len(map_files)-1
            model_dataset = '_'.join(map_files[0].split('_')[:-1])
            # print(os.path.join(self.map_dir,f'{model_dataset}_{cur_map}.npy'))
            self.saliency_map_latest = np.load(os.path.join(self.map_dir,f'{model_dataset}_{cur_map}.npy'))# each map is binary
            
            # self.saliency_map = [np.load(os.path.join(self.map_dir,f_path)) for f_path in map_files] # each map is binary
            self.saliency_map = self.saliency_map_latest
            # print(np.stack(self.saliency_map).shape)
            # self.saliency_map = np.stack(self.saliency_map).max(axis=0)
            # print(self.saliency_map.shape)

    def _check_legacy_exist(self):
        processed_folder_exists = os.path.exists(self.processed_folder)
        if not processed_folder_exists:
            return False

        return all(
            check_integrity(os.path.join(self.processed_folder, file)) for file in (self.training_file, self.test_file)
        )

    def _load_legacy_data(self):
        # This is for BC only. We no longer cache the data in a custom binary, but simply read from the raw data
        # directly.
        data_file = self.training_file if self.train else self.test_file
        return torch.load(os.path.join(self.processed_folder, data_file))

    def _load_data(self):
        image_file = f"{'train' if self.train else 't10k'}-images-idx3-ubyte"
        data = read_image_file(os.path.join(self.raw_folder, image_file))

        label_file = f"{'train' if self.train else 't10k'}-labels-idx1-ubyte"
        targets = read_label_file(os.path.join(self.raw_folder, label_file))

        return data, targets

    # def __getitem__(self, index: int) -> Tuple[Any, Any]:
    #     """
    #     Args:
    #         index (int): Index

    #     Returns:
    #         tuple: (image, target) where target is index of the target class.
    #     """
    #     img, target = self.data[index], int(self.targets[index])

    #     # doing this so that it is consistent with all other datasets
    #     # to return a PIL Image
    #     img = Image.fromarray(img.numpy(), mode='L')

    #     if self.transform is not None:
    #         img = self.transform(img)

    #     if self.target_transform is not None:
    #         target = self.target_transform(target)

    #     return img, target

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        
        # img  = np.expand_dims(img,-1)
        
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if self.use_basemap:
            if self.base_map is not None:
                saliency_map = np.expand_dims(1/(1 + np.exp(-self.base_map[index])),-1)
                img = img*(1-saliency_map)
                img = ((img-img.min())/(img.max()-img.min())*255)
            
            img = Image.fromarray(img.astype(np.uint8))
            
        elif self.saliency_map is not None:
            if self.use_latest_map:
                saliency_map = self.saliency_map_latest[index]
            else:
                saliency_map = self.saliency_map[index]
            if self.bootstrap_maps:
                num_boots = 10
                img = [Image.fromarray(erase_by_map(img,saliency_map,self.smooth_map,self.map_out,kernel_size = self.kernel_size,sigma=self.sigma,noisey=True).astype(np.uint8)) for _ in range(num_boots)]
            else:
                img = erase_by_map(img,saliency_map,self.smooth_map,self.map_out,kernel_size = self.kernel_size,sigma=self.sigma,noisey = self.pertub!='cmean')
                # print(saliency_map.sum(),saliency_map.shape)
                img = Image.fromarray(img.astype(np.uint8))
        else:
            # print(img.shape,'-----')
            img = Image.fromarray(img.astype(np.uint8))

        if self.transform is not None:
            if self.bootstrap_maps and self.saliency_map is not None:
                img = [self.transform(i) for i in img]
            else:
                img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        # print(img.size())
        return img, target

    def __len__(self) -> int:
        return len(self.data)

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def class_to_idx(self) -> Dict[str, int]:
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self) -> bool:
        return all(
            check_integrity(os.path.join(self.raw_folder, os.path.splitext(os.path.basename(url))[0]))
            for url, _ in self.resources
        )

    def download(self) -> None:
        """Download the MNIST data if it doesn't exist already."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)

        # download files
        for filename, md5 in self.resources:
            for mirror in self.mirrors:
                url = "{}{}".format(mirror, filename)
                try:
                    print("Downloading {}".format(url))
                    download_and_extract_archive(
                        url, download_root=self.raw_folder,
                        filename=filename,
                        md5=md5
                    )
                except URLError as error:
                    print(
                        "Failed to download (trying next):\n{}".format(error)
                    )
                    continue
                finally:
                    print()
                break
            else:
                raise RuntimeError("Error downloading {}".format(filename))

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")

    def generate_discrete_saliency_map_parallel(self,model,map_alg,map_path,threshold,drop_max,use_cuda):
        
        def gen_one_map(i,drop_max):
            # if os.path.exists(map_file):
            #     continue
                    # print('found_basemap')
            grayscale_cam = self.base_map[i]
            grayscale_cam =  np.expand_dims(grayscale_cam,0)
            if not drop_max:
                grayscale_cam = -grayscale_cam
            grayscale_cam = grayscale_cam - np.min(grayscale_cam)
            if self.saliency_map is not None:
                grayscale_cam = grayscale_cam * np.expand_dims(1-self.saliency_map[i],0)
            idx = np.dstack(np.unravel_index(np.argsort(grayscale_cam.ravel()), (28, 28)))
            
            ridx,cidx = np.split(idx[0,-int(28*28*threshold):,:],2,axis=-1)
            
            out_cam = np.zeros_like(grayscale_cam)
            out_cam[0][ridx,cidx] = 1
            if self.saliency_map is not None:
                out_cam = out_cam + self.saliency_map[i]
            return out_cam
        grayscale_cams = Parallel(n_jobs=24,backend='threading')(delayed(gen_one_map)(i,drop_max) for i in range(len(self.base_map)))
        grayscale_cams = np.concatenate(grayscale_cams,axis=0).astype(dtype=np.float32)
        
        np.save(f"{map_path}.npy",grayscale_cams)
        return

    def remove_maps(self,map_path):
        if os.path.exists(f"{map_path}.npy"):
            os.remove(f"{map_path}.npy")
            print(f'deleted {map_path}.npy')

    def generate_discrete_saliency_map(self,model,map_alg,map_path,threshold,drop_max,use_cuda):
        if not self.iter_map and self.base_map is not None:
            return self.generate_discrete_saliency_map_parallel(model,map_alg,map_path,threshold,drop_max,use_cuda)
        
        try:
            target_layers = model.get_target_layers()
        except:
            target_layers = [model.features[-1]]
        if self.base_map is None:
            map_alg = attribute_image_object(algorithm=map_alg,alg_args={'model':model,'layer':target_layers[0],'multiply_by_inputs':True})
        grayscale_cams = []
        baselines = torch.randn(20, 1, 28, 28)
        base_maps=[]
        if use_cuda:
            baselines = baselines.cuda()
        for i in tqdm(range(self.data.shape[0])):
            input_tensor, target_category = self.__getitem__(i)
            target_category = target_category.tolist()
            # print(target_category.tolist())
            input_tensor = torch.unsqueeze(input_tensor, 0)
            if use_cuda:
                input_tensor = input_tensor.cuda()
            if self.iter_map:
                # print('iter map')
                grayscale_cam = attribute_image_features(attr_obj=map_alg,channel_aggregation='all',
                                                        att_args={'inputs':input_tensor,
                                                                'baselines':baselines,
                                                                'target':target_category})
            else:
                # print('no iter map')
                if self.base_map is None:
                    grayscale_cam = attribute_image_features(attr_obj=map_alg,channel_aggregation='all',
                                                            att_args={'inputs':input_tensor,
                                                                    'baselines':baselines,
                                                                    'target':target_category})
                    
                    base_map = np.expand_dims(grayscale_cam,0)
                    base_maps.append(base_map)
                else:
                    # print('found_basemap')
                    grayscale_cam = self.base_map[i]
            
            grayscale_cam =  np.expand_dims(grayscale_cam,0)
            if not drop_max:
                grayscale_cam = -grayscale_cam
            grayscale_cam = grayscale_cam - np.min(grayscale_cam)
            if self.saliency_map is not None:
                grayscale_cam = grayscale_cam * np.expand_dims(1-self.saliency_map[i],0)
            idx = np.dstack(np.unravel_index(np.argsort(grayscale_cam.ravel()), (28, 28)))
            
            ridx,cidx = np.split(idx[0,-int(28*28*threshold):,:],2,axis=-1)
            out_cam = np.zeros_like(grayscale_cam)
            # print(ridx,cidx,out_cam.shape)
            out_cam[0][ridx,cidx] = 1
            if self.saliency_map is not None:
                out_cam = out_cam + self.saliency_map[i]
            grayscale_cams.append(out_cam)
        grayscale_cams = np.concatenate(grayscale_cams,axis=0)
        if len(base_maps)!=0:
            base_map_dir = os.path.join(self.map_dir,'base_map.npy')
            print(f'save to {base_map_dir}')
            base_maps = np.concatenate(base_maps,axis=0)
            np.save(os.path.join(self.map_dir,'base_map.npy'),base_maps)
        np.save(f"{map_path}.npy",grayscale_cams)
    def generate_discrete_random_saliency_map(self,map_path,threshold):
        dummy_imgs = np.random.rand(self.data.shape[0],28, 28)
        if self.saliency_map is not None:
            dummy_imgs = dummy_imgs * (1-self.saliency_map)
        grayscale_cams = []
        def gen_one_map(i):
            grayscale_cam = dummy_imgs[i]
            idx = np.dstack(np.unravel_index(np.argsort(grayscale_cam.ravel()), (28, 28)))
            ridx,cidx = np.split(idx[0,-int(28*28*threshold):,:],2,axis=-1)
            out_cam = np.zeros_like(grayscale_cam)
            out_cam[ridx,cidx] = 1
            if self.saliency_map is not None:
                out_cam = out_cam+self.saliency_map[i]

            # print(idx.shape,grayscale_cam.max())
            return out_cam
            # grayscale_cams.append(out_cam)
        grayscale_cams = Parallel(n_jobs=24,backend='threading')(delayed(gen_one_map)(i) for i in range(len(dummy_imgs)))
        grayscale_cams = np.stack(grayscale_cams,axis=0).astype(dtype=np.float32)
        
        np.save(f"{map_path}.npy",grayscale_cams)

class E_Food101(VisionDataset):
    """`The Food-101 Data Set <https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/>`_.

    The Food-101 is a challenging data set of 101 food categories, with 101'000 images.
    For each class, 250 manually reviewed test images are provided as well as 750 training images.
    On purpose, the training images were not cleaned, and thus still contain some amount of noise.
    This comes mostly in the form of intense colors and sometimes wrong labels. All images were
    rescaled to have a maximum side length of 512 pixels.


    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"train"`` (default) and ``"test"``.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again. Default is False.
    """

    _URL = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
    _MD5 = "85eeb15f3717b99a5da872d97d918f87"

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        map_files: list = None,
        map_dir: str = None,
        map_out: bool = True,
        smooth_map: bool = True,
        kernel_size: int = 5,
        sigma: float = 5.0,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        # self.split = split  # training set or test set
        self.map_out = map_out
        self.smooth_map = smooth_map
        self.kernel_size = kernel_size
        self.sigma = sigma

        self._split = 'train' if split in ['train','val'] else 'test'
        self._split = verify_str_arg(self._split, "split", ("train", "test"))
        self._base_folder = Path(self.root) / "food-101"
        self._meta_folder = self._base_folder / "meta"
        self._images_folder = self._base_folder / "images"
        self.split = self._split # for consistance between dataloader
        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self._labels = []
        self._image_files = []
        with open(self._meta_folder / f"{self._split}.json") as f:
            metadata = json.loads(f.read())
        
        self.classes = sorted(metadata.keys())
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))

        for class_label, im_rel_paths in metadata.items():
            self._labels += [self.class_to_idx[class_label]] * len(im_rel_paths)
            self._image_files += [
                self._images_folder.joinpath(*f"{im_rel_path}.jpg".split("/")) for im_rel_path in im_rel_paths
            ]
        
        val_idx = np.load(Path(self.root)/ 'val_idx.npy')
        if split == 'val':
            self._labels = np.array(self._labels)[val_idx]
            self._image_files = np.array(self._image_files)[val_idx]
        elif split == 'train':
            self._labels = np.array(self._labels)[np.invert(val_idx)]
            self._image_files = np.array(self._image_files)[np.invert(val_idx)]
        
        self.saliency_map = None
        if map_files is not None and map_files!=[]:
            print('Found saliency_map')
            map_dir = os.path.join(map_dir,self.split)
            self.saliency_map = [[os.path.join(map_dir,f_path,f'{im.stem}.npy') for im in self._image_files] for f_path in map_files] # each map is binary
            self.saliency_map = np.array(self.saliency_map)


    def get_combined_map(self,index):
        map_files = self.saliency_map[:,index].flatten()
        # print(map_files,self._image_files[index])

        maps = np.array([np.load(f) for f in map_files]).max(axis=0)
        return maps

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        image_file, label = self._image_files[idx], self._labels[idx]
        image = Image.open(image_file).convert("RGB")
        if self.saliency_map is not None:
            saliency_map = self.get_combined_map(idx)
            # print(saliency_map.shape)
            image = np.array(image)
            image = erase_by_map(image,saliency_map,self.smooth_map,self.map_out,kernel_size=self.kernel_size,sigma=self.sigma,noisey = self.pertub!='cmean')

            image = Image.fromarray(image.astype(np.uint8))
        
        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)
        return image, label


    def extra_repr(self) -> str:
        return f"split={self._split}"

    def _check_exists(self) -> bool:
        return all(folder.exists() and folder.is_dir() for folder in (self._meta_folder, self._images_folder))

    def _download(self) -> None:
        if self._check_exists():
            return
        download_and_extract_archive(self._URL, download_root=self.root, md5=self._MD5)
    
    def generate_discrete_saliency_map(self,model,cam,map_path,threshold,drop_max,use_cuda):
        
        target_layers = model.get_target_layers()
        # model.double()
        cam = cam(model=model, target_layers=target_layers, use_cuda=use_cuda)
        # grayscale_cams = []
        if not os.path.exists(map_path):
            os.mkdir(map_path)
        for i in tqdm(range(len(self._labels))):
            input_file_name = self._image_files[i].stem
            map_file = os.path.join(map_path,f'{input_file_name}.npy')
            if os.path.exists(map_file):
                continue
            input_tensor, target_category = self.__getitem__(i)
            input_tensor = torch.unsqueeze(input_tensor, 0)
            grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(target_category)])
            if self.saliency_map is not None:
                grayscale_cam = grayscale_cam * (1-self.get_combined_map(i))
            idx = np.dstack(np.unravel_index(np.argsort(grayscale_cam.ravel()), (224, 224)))
            ridx,cidx = np.split(idx[0,(-1*drop_max)*int(50176*threshold):,:],2,axis=-1)
            out_cam = np.zeros_like(grayscale_cam)
            out_cam[0][ridx,cidx] = 1
            
            np.save(map_file,out_cam)
            # grayscale_cams.append(out_cam)
        # grayscale_cams = np.concatenate(grayscale_cams,axis=0)
        
    def generate_discrete_random_saliency_map(self,map_path,threshold):
        if not os.path.exists(map_path):
            os.mkdir(map_path)
        for i in tqdm(range(len(self._labels))):
            input_file_name = self._image_files[i].stem
            map_file = os.path.join(map_path,f'{input_file_name}.npy')
            if os.path.exists(map_file):
                continue
            grayscale_cam = np.random.rand(224,224)
            if self.saliency_map is not None:
                grayscale_cam = grayscale_cam * (1-self.get_combined_map(i))
            idx = np.dstack(np.unravel_index(np.argsort(grayscale_cam.ravel()), (224, 224)))
            ridx,cidx = np.split(idx[0,-int(50176*threshold):,:],2,axis=-1)
            out_cam = np.zeros_like(grayscale_cam)
            out_cam[ridx,cidx] = 1
            # print(idx.shape,grayscale_cam.max())
            np.save(map_file,out_cam)
        # grayscale_cams = np.stack(grayscale_cams,axis=0).astype(dtype=np.float32)
        
        

class E_CIFAR100(E_CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }



ARCHIVE_META = {
    'train': ('ILSVRC2012_img_train.tar', '1d675b47d978889d74fa0da5fadfb00e'),
    'val': ('ILSVRC2012_img_val.tar', '29b22e2961454d5413ddabcf34fc5622'),
    'devkit': ('ILSVRC2012_devkit_t12.tar.gz', 'fa75699e90414af021442c21a62c3abf')
}

META_FILE = "meta.bin"

class E_ImageNet(ImageFolder):
    """`ImageNet <http://image-net.org/>`_ 2012 Classification Dataset.

    Args:
        root (string): Root directory of the ImageNet Dataset.
        split (string, optional): The dataset split, supports ``train``, or ``val``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class name tuples.
        class_to_idx (dict): Dict with items (class_name, class_index).
        wnids (list): List of the WordNet IDs.
        wnid_to_idx (dict): Dict with items (wordnet_id, class_index).
        imgs (list): List of (image path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, 
                 root: str, 
                 split: str = 'train', 
                 map_files: list = None,
                 map_dir: str = None,
                 map_out: bool = True,
                 iter_map: bool = True, # ignored if train
                 smooth_map: bool = True,
                 kernel_size: int = 5,
                 sigma: float = 5.0,
                 bootstrap_maps: bool = False,
                 use_latest_map: bool = False,
                 use_basemap:bool = False,
                 num_samples: int = 5040,
                 cur_iter: int = None,
                 pertub: str = 'cmean',
                 download: Optional[str] = None, **kwargs: Any) -> None:
        if download is True:
            msg = ("The dataset is no longer publicly accessible. You need to "
                   "download the archives externally and place them in the root "
                   "directory.")
            raise RuntimeError(msg)
        elif download is False:
            msg = ("The use of the download flag is deprecated, since the dataset "
                   "is no longer publicly accessible.")
            warnings.warn(msg, RuntimeWarning)

        root = self.root = os.path.expanduser(root)
        self.split = verify_str_arg(split, "split", ("train", "val"))
        self.map_out = map_out
        self.smooth_map = smooth_map
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.pertub = pertub
        self.bootstrap_maps = bootstrap_maps
        self.use_latest_map = use_latest_map
        self.use_basemap = use_basemap
        self.iter_map = iter_map if split in ['test','val'] else True
        self.cur_iter = cur_iter
        if map_dir is not None:
            self.map_dir = os.path.join(map_dir,self.split)
        self.parse_archives()
        wnid_to_classes = load_meta_file(self.root)[0]

        super(E_ImageNet, self).__init__(self.split_folder, **kwargs)
        self.root = root

        self.wnids = self.classes
        self.wnid_to_idx = self.class_to_idx
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        self.class_to_idx = {cls: idx
                             for idx, clss in enumerate(self.classes)
                             for cls in clss}
        self.samples = self.samples[:num_samples]
        self.base_map = None
        if not iter_map:
            print(os.path.join(self.map_dir,'base_map'))
            if os.path.exists(os.path.join(self.map_dir,'base_map',f'{Path(self.samples[0][0]).stem}.npy')):
                self.base_map = [os.path.join(self.map_dir,'base_map',f'{Path(im[0]).stem}.npy') for im in self.samples] # each map is binary
        self.saliency_map = None

        # does not use saliency map if it is the 0th iter
        if (map_files is not None and map_files!=[]) and self.cur_iter > 0:
            print('Found saliency_map','cur_iter:',self.cur_iter)
            map_dir = os.path.join(map_dir,self.split)
            
            if self.cur_iter is not None:
                # use cur_iter if given
                f_path = '_'.join(map_files[0].split('_')[:-1])+'_'+str(self.cur_iter-1)
            else:
                # else use the last map_file
                f_path = '_'.join(map_files[0].split('_')[:-1])+'_'+str(len(map_files)-1)
            self.saliency_map = [os.path.join(map_dir,f_path,f'{Path(im[0]).stem}.npy') for im in self.samples] # each map is binary
            # self.saliency_map = np.array(self.saliency_map)

    def parse_archives(self) -> None:
        if not check_integrity(os.path.join(self.root, META_FILE)):
            parse_devkit_archive(self.root)

        if not os.path.isdir(self.split_folder):
            if self.split == 'train':
                parse_train_archive(self.root)
            elif self.split == 'val':
                parse_val_archive(self.root)

    @property
    def split_folder(self) -> str:
        return os.path.join(self.root, self.split)

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        # sample = np.array(sample)
        if self.transform is not None:
            sample = self.transform(sample).numpy()
        # print(sample.shape)
        if self.use_basemap:
            if self.base_map is not None:
                saliency_map = np.expand_dims(1/(1 + np.exp(-self.base_map[index])),-1)
                sample = sample*(1-saliency_map)
            #     sample = ((sample-sample.min())/(sample.max()-sample.min())*255)
            # sample = Image.fromarray(sample.astype(np.uint8))
        elif self.saliency_map is not None:
            if self.use_latest_map:
                saliency_map = self.saliency_map_latest[index]
            else:
                saliency_map = self.get_combined_map(index)
            if self.bootstrap_maps:
                num_boots = 10
                sample = [Image.fromarray(erase_by_map(sample,saliency_map,self.smooth_map,self.map_out,kernel_size = self.kernel_size,sigma=self.sigma,noisey=True,channel_last=False).astype(np.uint8)) for _ in range(num_boots)]
            else:
                sample = erase_by_map(sample,saliency_map,self.smooth_map,self.map_out,kernel_size = self.kernel_size,sigma=self.sigma,channel_last=False,noisey = self.pertub!='cmean')
                # sample = Image.fromarray(sample.astype(np.uint8))
        # else:
        #     sample = Image.fromarray(sample.astype(np.uint8))
        sample = torch.from_numpy(sample)

        
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def get_combined_map(self,index):
        map_files = self.saliency_map[index]
        # print(map_files,self._image_files[index])

        maps = np.load(map_files)
        return maps

    def get_basemap(self,index):
        map_file = self.base_map[index]
        return np.load(map_file)
        
    def remove_maps(self,map_path):
        if os.path.exists(map_path):
            shutil.rmtree(map_path)
            print(f'deleted {map_path}')

    def generate_discrete_saliency_map(self,model,map_alg,map_path,threshold,drop_max,use_cuda):
        if not self.iter_map and self.base_map is not None:
            return self.generate_discrete_saliency_map_parallel(model,map_alg,map_path,threshold,drop_max,use_cuda)
        if not os.path.exists(map_path):
            os.mkdir(map_path)
        try:
            target_layers = model.get_target_layers()
        except:
            target_layers = [model.layer4[-1]]
        if self.base_map is None:
            map_alg = attribute_image_object(algorithm=map_alg,alg_args={'model':model,'layer':target_layers[0],'multiply_by_inputs':True})
        grayscale_cams = []
        
        baselines = torch.randn(10, 3, 224, 224)
        base_maps=[]
        base_map_path = os.path.join(self.map_dir,'base_map')
        if not os.path.exists(base_map_path):
            os.mkdir(base_map_path)
        if use_cuda:
            baselines = baselines.cuda()
        for i in tqdm(range(len(self.samples))):
            combined_map = None
            input_tensor, target_category = self.__getitem__(i)
            input_file_name = Path(self.samples[i][0]).stem
            map_file = os.path.join(map_path,f'{input_file_name}.npy')
            if os.path.exists(map_file):
                continue
            input_tensor = torch.unsqueeze(input_tensor, 0)
            if use_cuda:
                input_tensor = input_tensor.cuda()
            if self.iter_map:
                # print('iter map')
                grayscale_cam = attribute_image_features(attr_obj=map_alg,channel_aggregation='all',
                                                        att_args={'inputs':input_tensor,
                                                                'baselines':baselines,
                                                                'target':target_category})
            else:
                # print('no iter map')
                if self.base_map is None:
                    grayscale_cam = attribute_image_features(attr_obj=map_alg,channel_aggregation='all',
                                                            att_args={'inputs':input_tensor,
                                                                    'baselines':baselines,
                                                                    'target':target_category})
                    # base_map = np.expand_dims(grayscale_cam,0)
                    
                    base_map_file = os.path.join(base_map_path,f'{input_file_name}.npy')
                    np.save(base_map_file,grayscale_cam)
                else:
                    # print('found_basemap')
                    grayscale_cam = self.get_basemap(i)
            # grayscale_cam = ((grayscale_cam-grayscale_cam.min())/(grayscale_cam.max()-grayscale_cam.min())*255)
            grayscale_cam = Image.fromarray(grayscale_cam)
            grayscale_cam = np.array(grayscale_cam.resize((224,224)))
            grayscale_cam =  np.expand_dims(grayscale_cam,0)
            if not drop_max:
                grayscale_cam = -grayscale_cam
            grayscale_cam = grayscale_cam - np.min(grayscale_cam)
            print('drop_max:',drop_max)
            if self.saliency_map is not None:
                combined_map = self.get_combined_map(i)
                combined_map_enlarged = ndimage.maximum_filter(combined_map, size=9) # ensure non overlaping
                grayscale_cam = grayscale_cam * np.expand_dims(1-combined_map_enlarged,0)
            # idx = np.dstack(np.unravel_index(np.argsort(grayscale_cam.ravel()), (25,25)))
            
            # ridx,cidx = np.split(idx[0,-int(25*25*threshold):,:],2,axis=-1)
            ridx,cidx = np.unravel_index(np.squeeze(grayscale_cam).argmax(), np.squeeze(grayscale_cam).shape)
            # print(ridx,cidx)
            
            out_cam = np.zeros_like(grayscale_cam)
            out_cam[0][ridx,cidx] = 1
            out_cam = ndimage.maximum_filter(out_cam, size=9)
            if combined_map is not None:
                out_cam = out_cam = np.maximum(out_cam,combined_map)
            np.save(map_file,out_cam[0])
            # grayscale_cams.append(out_cam)
        # grayscale_cams = np.concatenate(grayscale_cams,axis=0)
    def generate_discrete_random_saliency_map(self,map_path,threshold):
        if not os.path.exists(map_path):
            os.mkdir(map_path)
        def gen_one_map(i):
            combined_map = None
            input_file_name = Path(self.samples[i][0]).stem
            map_file = os.path.join(map_path,f'{input_file_name}.npy')
            if os.path.exists(map_file):
                return i
            grayscale_cam = np.random.rand(224,224)
            if self.saliency_map is not None:
                # map_file = map_files[0].replace(map_files[0].split('/')[-2],f'ResNet50_E_ImageNet_{len(map_files)-1}')
                map_files = self.saliency_map[i]
                combined_map = np.load(map_files)
                grayscale_cam = grayscale_cam * (1-combined_map)
            idx = np.dstack(np.unravel_index(np.argsort(grayscale_cam.ravel()), (224,224)))
            ridx,cidx = np.split(idx[0,-int(224*224*threshold):,:],2,axis=-1)
            out_cam = np.zeros_like(grayscale_cam)
            out_cam[ridx,cidx] = 1
            
            # print(idx.shape,grayscale_cam.max())
            if combined_map is not None:
                out_cam = np.maximum(out_cam,combined_map)
            np.save(map_file,out_cam)
            return i
        r = Parallel(n_jobs=24,backend='threading')(delayed(gen_one_map)(i) for i in range(len(self.samples)))
        return r

    def generate_discrete_saliency_map_parallel(self,model,map_alg,map_path,threshold,drop_max,use_cuda):
        if not os.path.exists(map_path):
            os.mkdir(map_path)
        grayscale_cams = []
        base_maps=[]
        base_map_path = os.path.join(self.map_dir,'base_map')
        if not os.path.exists(base_map_path):
            os.mkdir(base_map_path)
        print('drop_max:',drop_max)
        def gen_one_map(i,drop_max, map_files, base_map_file):
            # return ireturn i
            input_file_name = Path(self.samples[i][0]).stem
            map_file = os.path.join(map_path,f'{input_file_name}.npy')
            if os.path.exists(map_file):
                return i
                    # print('found_basemap')
            grayscale_cam = np.load(base_map_file)
            combined_map = None
            # grayscale_cam = ((grayscale_cam-grayscale_cam.min())/(grayscale_cam.max()-grayscale_cam.min())*255)
            grayscale_cam = Image.fromarray(grayscale_cam)
            grayscale_cam =  np.expand_dims(grayscale_cam,0)
            if not drop_max:
                
                grayscale_cam = -grayscale_cam
            grayscale_cam = grayscale_cam - np.min(grayscale_cam) # - min because will replace previous mask by 0
            if self.saliency_map is not None:
                # map_file = '_'.join(map_files.split('_')[:-1])+'_'+str(len(map_files)-1)
                # map_file = map_files[0].replace(map_files[0].split('/')[-2],f'ResNet50_E_ImageNet_{len(map_files)-1}')
                combined_map = np.load(self.saliency_map[i])
                combined_map_enlarged = ndimage.maximum_filter(combined_map, size=9) # ensure non overlaping
                grayscale_cam = grayscale_cam * np.expand_dims(1-combined_map_enlarged,0)
            
            # idx = np.dstack(np.unravel_index(np.argsort(grayscale_cam.ravel()), (25, 25)))
            ridx,cidx = np.unravel_index(np.squeeze(grayscale_cam).argmax(), np.squeeze(grayscale_cam).shape)
            # print(ridx,cidx)
            # ridx,cidx = np.split(idx[0,-int(25*25*threshold):,:],2,axis=-1)
            
            out_cam = np.zeros_like(grayscale_cam,dtype=np.int32)

            out_cam[0][ridx,cidx] = 1
            out_cam = ndimage.maximum_filter(out_cam, size=9)
            # print(np.sum(out_cam>0))
            if combined_map is not None:
                out_cam = np.maximum(out_cam,combined_map)
            np.save(map_file,out_cam[0])
            return i
        
        r = Parallel(n_jobs=24,backend='threading')(delayed(gen_one_map)(i,drop_max,self.saliency_map,self.base_map[i]) for i in range(len(self.samples)))
        return r

def load_meta_file(root: str, file: Optional[str] = None) -> Tuple[Dict[str, str], List[str]]:
    if file is None:
        file = META_FILE
    file = os.path.join(root, file)

    if check_integrity(file):
        return torch.load(file)
    else:
        msg = ("The meta file {} is not present in the root directory or is corrupted. "
               "This file is automatically created by the ImageNet dataset.")
        raise RuntimeError(msg.format(file, root))


def _verify_archive(root: str, file: str, md5: str) -> None:
    if not check_integrity(os.path.join(root, file), md5):
        msg = ("The archive {} is not present in the root directory or is corrupted. "
               "You need to download it externally and place it in {}.")
        raise RuntimeError(msg.format(file, root))


def parse_devkit_archive(root: str, file: Optional[str] = None) -> None:
    """Parse the devkit archive of the ImageNet2012 classification dataset and save
    the meta information in a binary file.

    Args:
        root (str): Root directory containing the devkit archive
        file (str, optional): Name of devkit archive. Defaults to
            'ILSVRC2012_devkit_t12.tar.gz'
    """
    import scipy.io as sio

    def parse_meta_mat(devkit_root: str) -> Tuple[Dict[int, str], Dict[str, str]]:
        metafile = os.path.join(devkit_root, "data", "meta.mat")
        meta = sio.loadmat(metafile, squeeze_me=True)['synsets']
        nums_children = list(zip(*meta))[4]
        meta = [meta[idx] for idx, num_children in enumerate(nums_children)
                if num_children == 0]
        idcs, wnids, classes = list(zip(*meta))[:3]
        classes = [tuple(clss.split(', ')) for clss in classes]
        idx_to_wnid = {idx: wnid for idx, wnid in zip(idcs, wnids)}
        wnid_to_classes = {wnid: clss for wnid, clss in zip(wnids, classes)}
        return idx_to_wnid, wnid_to_classes

    def parse_val_groundtruth_txt(devkit_root: str) -> List[int]:
        file = os.path.join(devkit_root, "data",
                            "ILSVRC2012_validation_ground_truth.txt")
        with open(file, 'r') as txtfh:
            val_idcs = txtfh.readlines()
        return [int(val_idx) for val_idx in val_idcs]

    @contextmanager
    def get_tmp_dir() -> Iterator[str]:
        tmp_dir = tempfile.mkdtemp()
        try:
            yield tmp_dir
        finally:
            shutil.rmtree(tmp_dir)

    archive_meta = ARCHIVE_META["devkit"]
    if file is None:
        file = archive_meta[0]
    md5 = archive_meta[1]

    _verify_archive(root, file, md5)

    with get_tmp_dir() as tmp_dir:
        extract_archive(os.path.join(root, file), tmp_dir)

        devkit_root = os.path.join(tmp_dir, "ILSVRC2012_devkit_t12")
        idx_to_wnid, wnid_to_classes = parse_meta_mat(devkit_root)
        val_idcs = parse_val_groundtruth_txt(devkit_root)
        val_wnids = [idx_to_wnid[idx] for idx in val_idcs]

        torch.save((wnid_to_classes, val_wnids), os.path.join(root, META_FILE))


def parse_train_archive(root: str, file: Optional[str] = None, folder: str = "train") -> None:
    """Parse the train images archive of the ImageNet2012 classification dataset and
    prepare it for usage with the ImageNet dataset.

    Args:
        root (str): Root directory containing the train images archive
        file (str, optional): Name of train images archive. Defaults to
            'ILSVRC2012_img_train.tar'
        folder (str, optional): Optional name for train images folder. Defaults to
            'train'
    """
    archive_meta = ARCHIVE_META["train"]
    if file is None:
        file = archive_meta[0]
    md5 = archive_meta[1]

    _verify_archive(root, file, md5)

    train_root = os.path.join(root, folder)
    extract_archive(os.path.join(root, file), train_root)

    archives = [os.path.join(train_root, archive) for archive in os.listdir(train_root)]
    for archive in archives:
        extract_archive(archive, os.path.splitext(archive)[0], remove_finished=True)


def parse_val_archive(
    root: str, file: Optional[str] = None, wnids: Optional[List[str]] = None, folder: str = "val"
) -> None:
    """Parse the validation images archive of the ImageNet2012 classification dataset
    and prepare it for usage with the ImageNet dataset.

    Args:
        root (str): Root directory containing the validation images archive
        file (str, optional): Name of validation images archive. Defaults to
            'ILSVRC2012_img_val.tar'
        wnids (list, optional): List of WordNet IDs of the validation images. If None
            is given, the IDs are loaded from the meta file in the root directory
        folder (str, optional): Optional name for validation images folder. Defaults to
            'val'
    """
    archive_meta = ARCHIVE_META["val"]
    if file is None:
        file = archive_meta[0]
    md5 = archive_meta[1]
    if wnids is None:
        wnids = load_meta_file(root)[1]

    _verify_archive(root, file, md5)

    val_root = os.path.join(root, folder)
    extract_archive(os.path.join(root, file), val_root)

    images = sorted([os.path.join(val_root, image) for image in os.listdir(val_root)])

    for wnid in set(wnids):
        os.mkdir(os.path.join(val_root, wnid))

    for wnid, img_file in zip(wnids, images):
        shutil.move(img_file, os.path.join(val_root, wnid, os.path.basename(img_file)))
    

DATASETS = {
    'E_CIFAR10':E_CIFAR10,
    'E_Food101':E_Food101,
    'E_MNIST':E_MNIST,
    'E_ImageNet':E_ImageNet
}

def get_dataset(dataset_name):
    return DATASETS[dataset_name]


if __name__=='__main__':
    
    transform = transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.PILToTensor(),
                                        transforms.ConvertImageDtype(torch.float),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ])
    # train_dataset = E_ImageNet('./data/ImageNet',transform=transform,split='val')
    # r = train_dataset.generate_discrete_saliency_map_parallel(model=1,map_alg=1,map_path=1,threshold=1,drop_max=1,use_cuda=1)
    # print(r)
    # # exit()
    # transform=transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.1307,), (0.3081,))
    #     ])
    map_files=[f'SimpleCNN_E_MNIST_{i}.npy' for i in range(100)]#,'ResNet18_E_CIFAR10_1.npy']#,,'ResNet18_E_CIFAR10_2.npy''ResNet18_E_CIFAR10_3.npy','ResNet18_E_CIFAR10_4.npy','ResNet18_E_CIFAR10_5.npy','ResNet18_E_CIFAR10_7.npy','ResNet18_E_CIFAR10_8.npy','ResNet18_E_CIFAR10_9.npy']#,'vgg16_E_CIFAR10_3.npy','vgg16_E_CIFAR10_4.npy','vgg16_E_CIFAR10_5.npy','vgg16_E_CIFAR10_6.npy','vgg16_E_CIFAR10_7.npy','vgg16_E_CIFAR10_8.npy','vgg16_E_CIFAR10_9.npy']
    map_files=[f'vgg16_E_CIFAR10_{i}.npy' for i in range(100)]#,'ResNet18_E_CIFAR10_1.npy']#,,'ResNet18_E_CIFAR10_2.npy''ResNet18_E_CIFAR10_3.npy','ResNet18_E_CIFAR10_4.npy','ResNet18_E_CIFAR10_5.npy','ResNet18_E_CIFAR10_7.npy','ResNet18_E_CIFAR10_8.npy','ResNet18_E_CIFAR10_9.npy']#,'vgg16_E_CIFAR10_3.npy','vgg16_E_CIFAR10_4.npy','vgg16_E_CIFAR10_5.npy','vgg16_E_CIFAR10_6.npy','vgg16_E_CIFAR10_7.npy','vgg16_E_CIFAR10_8.npy','vgg16_E_CIFAR10_9.npy']
    
    # map_files = None/home/yimupan/ECCV2022/SaliencyAnalysis/data/CIFAR10/vgg16_Random_test_100_saliency_maps
    # train_dataset = E_CIFAR10('./data/CIFAR10',transform=transform, split='test',smooth_map=False,kernel_size=3,sigma=0.5,use_basemap=False,iter_map=False,map_files=map_files,map_dir='/home/yimupan/ECCV2022/SaliencyAnalysis/data/CIFAR10/vgg16_GuidedBackprop_no_iter_test_100_saliency_maps_mean',download=True,map_out=True,pertub='cmean')
    # train_dataset = E_MNIST('./data/MNIST',transform=transform, split='test',smooth_map=False, use_basemap=False,kernel_size=3,sigma=3.0,iter_map=False,map_files=map_files,map_dir='/home/yimupan/ECCV2022/SaliencyAnalysis/data/MNIST/SimpleCNN_GuidedBackprop_no_iter_test_100_saliency_maps',download=True,map_out=True,pertub='cmean')
    map_files=[f'ResNet50_E_ImageNet_{i}' for i in range(200)]
    train_dataset = E_ImageNet('./data/ImageNet',transform=transform,split='val',smooth_map=False, use_basemap=False,iter_map=False,map_files=map_files,map_dir='/home/yimupan/ECCV2022/SaliencyAnalysis/data/ImageNet/ResNet50_GuidedBackprop_val_100_saliency_maps_LeRF',map_out=True,pertub='cmean',cur_iter=3)
    # # test_dataset = E_CIFAR10('./data/CIFAR10',split='test',download=True)
    # map_files = ['ResNet50_E_Food101_1','ResNet50_E_Food101_2','ResNet50_E_Food101_3','ResNet50_E_Food101_4','ResNet50_E_Food101_5']
    # map_files = ['ResNet50_E_Food101_0','ResNet50_E_Food101_1','ResNet50_E_Food101_2','ResNet50_E_Food101_3','ResNet50_E_Food101_4','ResNet50_E_Food101_5']
    # map_dir = '/home/yimupan/ECCV2022/SaliencyAnalysis/data/Food101/Random_test_saliency_maps'#'/home/yimupan/ECCV2022/SaliencyAnalysis/data/Food101/GradCAM_test_saliency_maps'
    # train_dataset = E_Food101('./data/Food101',split='test',map_files=map_files,map_dir=map_dir,smooth_map=True,transform=transform,download=True,map_out=True)
    # test_dataset = E_Food101('./data/Food101',split='test',transform=transform,download=True)
    img, lb = train_dataset[10]
    base_map = train_dataset.get_basemap(10)
    
    print(img.shape)
    img = img.cpu().detach().numpy().transpose((1, 2, 0))
    img_plt = plt.imshow(img*0.5+0.5)
    # plt.imshow((base_map-base_map.min())/(base_map.max()-base_map.min()),cmap='Greys')
    # s_map = train_dataset.base_map[110]
    # s_map = train_dataset.saliency_map.sum(0)
    # train_dataset.generate_discrete_random_saliency_map('./data/test.npy',0.05)
    # print(len(np.unique(s_map)))
    # img_plt = plt.imshow(np.squeeze(s_map))
    plt.show()
    # print(len(train_dataset._labels))
    # print(len(val_dataset._labels))
    # print(len(test_dataset._labels))
    # # idx =np.random.choice(len(train_dataset._labels),size=len(train_dataset._labels)//5,replace=False).astype(int)
    # # bool_idx = np.zeros(len(train_dataset._labels))
    # # bool_idx[idx]=1
    # # np.save('./data/Food101/val_idx.npy',bool_idx.astype(bool))
    # # print(le)
    # print(np.array([0]*len(train_dataset._labels))[idx])
    # idx =np.random.choice(len(train_dataset._labels),size=len(train_dataset._labels)//5,replace=False).astype(int)

   

