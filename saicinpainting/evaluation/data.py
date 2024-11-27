import glob
import logging
import os

import cv2
import PIL.Image as Image
import numpy as np
import h5py

from torch.utils.data import Dataset
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)

def load_image(fname, mode='RGB', return_orig=False):
    img = np.array(Image.open(fname).convert(mode))
    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))
    out_img = img.astype('float32') / 255
    if return_orig:
        return out_img, img
    else:
        return out_img


def load_depth_from_file(fname, return_orig=False):
    depth = np.load(fname)
    # TODO: How to normalize depth maps
    out_depth = depth / np.max(depth)
    if return_orig:
        return out_depth, depth
    else:
        return out_depth

def normalize_depth(disparity_map):
    disparityf = disparity_map.astype(np.float32)
    disparity_not_zero = disparityf > 0.0
    disparity_zero = disparityf == 0.0
    true_depth = np.zeros(disparityf.shape)
    A = 1.0
    B = 0.0005
    max_depth = 1.0
    true_depth[disparity_not_zero] = 1/(A+B*disparityf[disparity_not_zero])
    true_depth[disparity_zero] = max_depth

    min_depth,max_depth = true_depth.min(), true_depth.max()
    true_depth_normalized = (true_depth - min_depth) / (max_depth - min_depth)
    return true_depth_normalized


def load_depth_from_hdf5(hdf5_path, depth_path, return_orig=False):
    with h5py.File(hdf5_path, 'r') as hdf5_file:
        # Access the dataset
        dataset = hdf5_file[depth_path]
        depth_map_u16 = dataset[:]
        
        # Retrieve mean and std attributes
        if 'min' in dataset.attrs:
            depth_map_min = dataset.attrs['min']
        else:
            LOGGER.warning(f"Attribute 'min' not found in dataset {depth_path}")
            depth_map_min = 4.1578239029840205 # average min of the dataset
                
        if 'range' in dataset.attrs:
            depth_map_range = dataset.attrs['range']
        else:
            LOGGER.warning(f"Attribute 'range' not found in dataset {depth_path}")
            depth_map_range = 519.2757145688164 # average range of the dataset
        
        # Convert uint16 back to normalized float
        divisor = 65535.0 if depth_map_u16.dtype == np.uint16 else 255.0 
        depth_map_normalized = depth_map_u16.astype(np.float32) / divisor
        
        # Denormalize to reconstruct the original depth map
        depth = depth_map_normalized * depth_map_range + depth_map_min
        
        #out_depth = depth / np.max(depth)
        out_depth = normalize_depth(depth)

        out_depth = out_depth.astype(np.float32)
        if return_orig:
            return out_depth, depth
        else:
            return out_depth


def ceil_modulo(x, mod):
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod


def pad_img_to_modulo(img, mod):
    channels, height, width = img.shape
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return np.pad(img, ((0, 0), (0, out_height - height), (0, out_width - width)), mode='symmetric')


def pad_tensor_to_modulo(img, mod):
    batch_size, channels, height, width = img.shape
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return F.pad(img, pad=(0, out_width - width, 0, out_height - height), mode='reflect')


def scale_image(img, factor, interpolation=cv2.INTER_AREA):
    if img.shape[0] == 1:
        img = img[0]
    else:
        img = np.transpose(img, (1, 2, 0))

    img = cv2.resize(img, dsize=None, fx=factor, fy=factor, interpolation=interpolation)

    if img.ndim == 2:
        img = img[None, ...]
    else:
        img = np.transpose(img, (2, 0, 1))
    return img


class InpaintingDataset(Dataset):
    def __init__(self, datadir, img_suffix='.jpg', pad_out_to_modulo=None, scale_factor=None):
        self.datadir = datadir
        self.mask_filenames = sorted(list(glob.glob(os.path.join(self.datadir, '**', '*mask*.png'), recursive=True)))
        self.img_filenames = [fname.rsplit('_mask', 1)[0] + img_suffix for fname in self.mask_filenames]
        self.pad_out_to_modulo = pad_out_to_modulo
        self.scale_factor = scale_factor

    def __len__(self):
        return len(self.mask_filenames)

    def __getitem__(self, i):
        image = load_image(self.img_filenames[i], mode='RGB')
        mask = load_image(self.mask_filenames[i], mode='L')
        result = dict(image=image, mask=mask[None, ...])

        if self.scale_factor is not None:
            result['image'] = scale_image(result['image'], self.scale_factor)
            result['mask'] = scale_image(result['mask'], self.scale_factor, interpolation=cv2.INTER_NEAREST)

        if self.pad_out_to_modulo is not None and self.pad_out_to_modulo > 1:
            result['unpad_to_size'] = result['image'].shape[1:]
            result['image'] = pad_img_to_modulo(result['image'], self.pad_out_to_modulo)
            result['mask'] = pad_img_to_modulo(result['mask'], self.pad_out_to_modulo)

        return result

class OurInpaintingDataset(Dataset):
    def __init__(self, datadir, img_suffix='.jpg', pad_out_to_modulo=None, scale_factor=None):
        self.datadir = datadir
        self.mask_filenames = sorted(list(glob.glob(os.path.join(self.datadir, 'mask', '**', '*mask*.png'), recursive=True)))
        self.img_filenames = [os.path.join(self.datadir, 'img', os.path.basename(fname.rsplit('-', 1)[0].rsplit('_', 1)[0]) + '.png') for fname in self.mask_filenames]
        self.pad_out_to_modulo = pad_out_to_modulo
        self.scale_factor = scale_factor

    def __len__(self):
        return len(self.mask_filenames)

    def __getitem__(self, i):
        result = dict(image=load_image(self.img_filenames[i], mode='RGB'),
                      mask=load_image(self.mask_filenames[i], mode='L')[None, ...])

        if self.scale_factor is not None:
            result['image'] = scale_image(result['image'], self.scale_factor)
            result['mask'] = scale_image(result['mask'], self.scale_factor)

        if self.pad_out_to_modulo is not None and self.pad_out_to_modulo > 1:
            result['image'] = pad_img_to_modulo(result['image'], self.pad_out_to_modulo)
            result['mask'] = pad_img_to_modulo(result['mask'], self.pad_out_to_modulo)

        return result

class DepthInpaintingEvaluationDataset(Dataset):
    def __init__(self, img_datadir, depth_datadir, img_suffix='.jpg', depth_suffix='.npy', pad_out_to_modulo=None, scale_factor=None):
        self.img_datadir = img_datadir
        self.depth_datadir = depth_datadir
        self.mask_filenames = sorted(list(glob.glob(os.path.join(self.img_datadir, '**', '*mask*.*'), recursive=True)))
        self.img_filenames = [fname.rsplit('_mask', 1)[0] + img_suffix for fname in self.mask_filenames]
        # TODO: How are the depth maps named in the directory
        self.depth_filenames = [os.path.join(self.depth_datadir,
            os.path.basename(fname).rsplit('_mask', 1)[0] + depth_suffix) for fname in self.mask_filenames]
        self.pad_out_to_modulo = pad_out_to_modulo
        self.scale_factor = scale_factor

    def __len__(self):
        return len(self.mask_filenames)

    def __getitem__(self, i):
        image = load_image(self.img_filenames[i], mode='RGB')
        mask = load_image(self.mask_filenames[i], mode='L')
        depth = load_depth_from_file(self.depth_filenames[i])
        result = dict(image=image, mask=mask[None, ...], depth=depth[None, ...])

        if self.scale_factor is not None:
            result['image'] = scale_image(result['image'], self.scale_factor)
            result['mask'] = scale_image(result['mask'], self.scale_factor, interpolation=cv2.INTER_NEAREST)
            result['depth'] = scale_image(result['depth'], self.scale_factor) # TODO: How to resize the depth maps

        if self.pad_out_to_modulo is not None and self.pad_out_to_modulo > 1:
            result['unpad_to_size'] = result['image'].shape[1:]
            result['image'] = pad_img_to_modulo(result['image'], self.pad_out_to_modulo)
            result['mask'] = pad_img_to_modulo(result['mask'], self.pad_out_to_modulo)
            result['depth'] = pad_img_to_modulo(result['depth'], self.pad_out_to_modulo)

        return result


class DepthInpaintingEvaluationWithHdf5Dataset(Dataset):
    def __init__(self, img_datadir, hdf5_path, img_suffix='.jpg', pad_out_to_modulo=None, scale_factor=None):
        self.img_datadir = img_datadir
        self.hdf5_path = hdf5_path
        self.mask_filenames = sorted(list(glob.glob(os.path.join(self.img_datadir, '**', '*mask*.*'), recursive=True)))
        self.img_filenames = [fname.rsplit('_mask', 1)[0] + img_suffix for fname in self.mask_filenames]
        depth_root = "val" if "/val" in img_datadir else "visual_test"
        self.depth_paths = [path.replace(img_datadir, depth_root).replace("\\","/").removesuffix(img_suffix) for path in self.img_filenames]
        self.pad_out_to_modulo = pad_out_to_modulo
        self.scale_factor = scale_factor

    def __len__(self):
        return len(self.mask_filenames)

    def __getitem__(self, i):
        image = load_image(self.img_filenames[i], mode='RGB')
        mask = load_image(self.mask_filenames[i], mode='L')
        depth = load_depth_from_hdf5(self.hdf5_path, self.depth_paths[i])
        result = dict(image=image, mask=mask[None, ...], depth=depth[None, ...])

        if self.scale_factor is not None:
            result['image'] = scale_image(result['image'], self.scale_factor)
            result['mask'] = scale_image(result['mask'], self.scale_factor, interpolation=cv2.INTER_NEAREST)
            result['depth'] = scale_image(result['depth'], self.scale_factor) # TODO: How to resize the depth maps

        if self.pad_out_to_modulo is not None and self.pad_out_to_modulo > 1:
            result['unpad_to_size'] = result['image'].shape[1:]
            result['image'] = pad_img_to_modulo(result['image'], self.pad_out_to_modulo)
            result['mask'] = pad_img_to_modulo(result['mask'], self.pad_out_to_modulo)
            result['depth'] = pad_img_to_modulo(result['depth'], self.pad_out_to_modulo)

        #LOGGER.info(f"Index {i}: img shape {result['image'].shape}, mask shape {result['mask'].shape}, depth shape {result['depth'].shape}")
        return result

class PrecomputedInpaintingResultsDataset(InpaintingDataset):
    def __init__(self, datadir, predictdir, inpainted_suffix='_inpainted.jpg', **kwargs):
        super().__init__(datadir, **kwargs)
        if not datadir.endswith('/'):
            datadir += '/'
        self.predictdir = predictdir
        self.pred_filenames = [os.path.join(predictdir, os.path.splitext(fname[len(datadir):])[0] + inpainted_suffix)
                               for fname in self.mask_filenames]

    def __getitem__(self, i):
        result = super().__getitem__(i)
        result['inpainted'] = load_image(self.pred_filenames[i])
        if self.pad_out_to_modulo is not None and self.pad_out_to_modulo > 1:
            result['inpainted'] = pad_img_to_modulo(result['inpainted'], self.pad_out_to_modulo)
        return result

class OurPrecomputedInpaintingResultsDataset(OurInpaintingDataset):
    def __init__(self, datadir, predictdir, inpainted_suffix="png", **kwargs):
        super().__init__(datadir, **kwargs)
        if not datadir.endswith('/'):
            datadir += '/'
        self.predictdir = predictdir
        self.pred_filenames = [os.path.join(predictdir, os.path.basename(os.path.splitext(fname)[0]) + f'_inpainted.{inpainted_suffix}')
                               for fname in self.mask_filenames]
        # self.pred_filenames = [os.path.join(predictdir, os.path.splitext(fname[len(datadir):])[0] + inpainted_suffix)
        #                        for fname in self.mask_filenames]

    def __getitem__(self, i):
        result = super().__getitem__(i)
        result['inpainted'] = self.file_loader(self.pred_filenames[i])

        if self.pad_out_to_modulo is not None and self.pad_out_to_modulo > 1:
            result['inpainted'] = pad_img_to_modulo(result['inpainted'], self.pad_out_to_modulo)
        return result

class InpaintingEvalOnlineDataset(Dataset):
    def __init__(self, indir, mask_generator, img_suffix='.jpg', pad_out_to_modulo=None, scale_factor=None,  **kwargs):
        self.indir = indir
        self.mask_generator = mask_generator
        self.img_filenames = sorted(list(glob.glob(os.path.join(self.indir, '**', f'*{img_suffix}' ), recursive=True)))
        self.pad_out_to_modulo = pad_out_to_modulo
        self.scale_factor = scale_factor

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, i):
        img, raw_image = load_image(self.img_filenames[i], mode='RGB', return_orig=True)
        mask = self.mask_generator(img, raw_image=raw_image)
        result = dict(image=img, mask=mask)

        if self.scale_factor is not None:
            result['image'] = scale_image(result['image'], self.scale_factor)
            result['mask'] = scale_image(result['mask'], self.scale_factor, interpolation=cv2.INTER_NEAREST)

        if self.pad_out_to_modulo is not None and self.pad_out_to_modulo > 1:
            result['image'] = pad_img_to_modulo(result['image'], self.pad_out_to_modulo)
            result['mask'] = pad_img_to_modulo(result['mask'], self.pad_out_to_modulo)
        return result
