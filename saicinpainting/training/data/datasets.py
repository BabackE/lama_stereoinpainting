import glob
import logging
import os
import random

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import webdataset
from omegaconf import open_dict, OmegaConf
from skimage.feature import canny
from skimage.transform import rescale, resize
from torch.utils.data import Dataset, IterableDataset, DataLoader, DistributedSampler, ConcatDataset

from saicinpainting.evaluation.data import InpaintingDataset as InpaintingEvaluationDataset, \
    OurInpaintingDataset as OurInpaintingEvaluationDataset, ceil_modulo, InpaintingEvalOnlineDataset, \
    DepthInpaintingEvaluationDataset, DepthInpaintingEvaluationWithHdf5Dataset, RGB565DInpaintingEvaluationWithHdf5Dataset, \
    load_depth_from_file, load_depth_from_hdf5
from saicinpainting.training.data.aug import IAAAffine2, IAAPerspective2
from saicinpainting.training.data.masks import get_mask_generator

LOGGER = logging.getLogger(__name__)


class InpaintingTrainDataset(Dataset):
    def __init__(self, indir, mask_generator, transform):
        self.in_files = list(glob.glob(os.path.join(indir, '**', '*.jpg'), recursive=True))
        self.mask_generator = mask_generator
        self.transform = transform
        self.iter_i = 0

    def __len__(self):
        return len(self.in_files)

    def __getitem__(self, item):
        path = self.in_files[item]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(image=img)['image']
        img = np.transpose(img, (2, 0, 1))
        # TODO: maybe generate mask before augmentations? slower, but better for segmentation-based masks
        mask = self.mask_generator(img, iter_i=self.iter_i)
        self.iter_i += 1
        return dict(image=img,
                    mask=mask)


class InpaintingTrainWebDataset(IterableDataset):
    def __init__(self, indir, mask_generator, transform, shuffle_buffer=200):
        self.impl = webdataset.Dataset(indir).shuffle(shuffle_buffer).decode('rgb').to_tuple('jpg')
        self.mask_generator = mask_generator
        self.transform = transform

    def __iter__(self):
        for iter_i, (img,) in enumerate(self.impl):
            img = np.clip(img * 255, 0, 255).astype('uint8')
            img = self.transform(image=img)['image']
            img = np.transpose(img, (2, 0, 1))
            mask = self.mask_generator(img, iter_i=iter_i)
            yield dict(image=img,
                       mask=mask)


class ImgSegmentationDataset(Dataset):
    def __init__(self, indir, mask_generator, transform, out_size, segm_indir, semantic_seg_n_classes):
        self.indir = indir
        self.segm_indir = segm_indir
        self.mask_generator = mask_generator
        self.transform = transform
        self.out_size = out_size
        self.semantic_seg_n_classes = semantic_seg_n_classes
        self.in_files = list(glob.glob(os.path.join(indir, '**', '*.jpg'), recursive=True))

    def __len__(self):
        return len(self.in_files)

    def __getitem__(self, item):
        path = self.in_files[item]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.out_size, self.out_size))
        img = self.transform(image=img)['image']
        img = np.transpose(img, (2, 0, 1))
        mask = self.mask_generator(img)
        segm, segm_classes= self.load_semantic_segm(path)
        result = dict(image=img,
                      mask=mask,
                      segm=segm,
                      segm_classes=segm_classes)
        return result

    def load_semantic_segm(self, img_path):
        segm_path = img_path.replace(self.indir, self.segm_indir).replace(".jpg", ".png")
        mask = cv2.imread(segm_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.out_size, self.out_size))
        tensor = torch.from_numpy(np.clip(mask.astype(int)-1, 0, None))
        ohe = F.one_hot(tensor.long(), num_classes=self.semantic_seg_n_classes) # w x h x n_classes
        return ohe.permute(2, 0, 1).float(), tensor.unsqueeze(0)


class DepthInpaintingTrainDataset(Dataset):
    def __init__(self, indir, depth_datadir, mask_generator, transform):
        self.in_files = list(glob.glob(os.path.join(indir, '**', '*.jpg'), recursive=True))
        # TODO: How are the depth files named in the directory
        self.depth_files = [path.split(indir,1)[1].replace("\\","/").removesuffix(".jpg") for path in self.in_files]
        self.mask_generator = mask_generator
        self.transform = transform
        self.iter_i = 0

    def __len__(self):
        return len(self.in_files)

    def __getitem__(self, item):
        path = self.in_files[item]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        depth = load_depth_from_file(self.depth_files[item])
        # TODO: is it fine to transform depth as a mask 
        transform_result = self.transform(image=img, mask=depth)
        img = transform_result['image']
        img = np.transpose(img, (2, 0, 1))
        depth = transform_result['mask']
        mask = self.mask_generator(img, iter_i=self.iter_i)
        self.iter_i += 1
        return dict(image=img,
                    mask=mask,
                    depth=depth)


class DepthInpaintingTrainWithHdf5Dataset(Dataset):
    def __init__(self, indir, hdf5_path, mask_generator, transform):
        self.in_files = list(glob.glob(os.path.join(indir, '**', '*.jpg'), recursive=True))
        self.hdf5_path = hdf5_path
        
        # TODO note:
        # read self.in_files and create equivalent depth path inside the hdf5 file that we'll look up in load_depth_from_hdf5

        self.depth_files = [path.split(indir,1)[1].replace("\\","/").removesuffix(".jpg") for path in self.in_files]
        self.mask_generator = mask_generator
        self.transform = transform
        self.iter_i = 0
        LOGGER.info(f"DEPTH DATALOADER WITH {len(self.in_files)} in {self.hdf5_path}")

    def __len__(self):
        return len(self.in_files)

    def __getitem__(self, item):
        path = self.in_files[item]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        depth = load_depth_from_hdf5(self.hdf5_path, self.depth_files[item])
        # TODO: is it fine to transform depth as a mask 
        transform_result = self.transform(image=img, mask=depth)
        img = transform_result['image']
        img = np.transpose(img, (2, 0, 1))
        depth = transform_result['mask']
        depth = depth[np.newaxis,...]
        depth = depth.astype('float32')
        mask = self.mask_generator(img, iter_i=self.iter_i)
        '''
        sample = {
            'img': img,        # e.g., (C, H, W)
            'mask': mask,      # e.g., (H, W)
            'depth': depth     # e.g., (H, W)
        }
        LOGGER.info(f"Index {item} {path}: img shape {sample['img'].shape}, mask shape {sample['mask'].shape}, depth shape {sample['depth'].shape}")
        '''
        self.iter_i += 1
        return dict(image=img,
                    mask=mask,
                    depth=depth)

class RGB565DInpaintingTrainWithHdf5Dataset(Dataset):
    def __init__(self, indir, hdf5_path, mask_generator, transform):
        self.in_files = list(glob.glob(os.path.join(indir, '**', '*.jpg'), recursive=True))
        self.hdf5_path = hdf5_path
        
        # TODO note:
        # read self.in_files and create equivalent depth path inside the hdf5 file that we'll look up in load_depth_from_hdf5

        self.depth_files = [path.split(indir,1)[1].replace("\\","/").removesuffix(".jpg") for path in self.in_files]
        self.mask_generator = mask_generator
        self.transform = transform
        self.iter_i = 0
        LOGGER.info(f"RGB565D DEPTH DATALOADER WITH {len(self.in_files)} in {self.hdf5_path}")

    def __len__(self):
        return len(self.in_files)

    def __getitem__(self, item):
        path = self.in_files[item]
        
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        depth = load_depth_from_hdf5(self.hdf5_path, self.depth_files[item])

        # TODO: is it fine to transform depth as a mask 
        transform_result = self.transform(image=img, mask=depth)
        img = transform_result['image']

        # convert transformed RGB image into BGR565
        img_u8 = (img * 255.0).astype('uint8')
        bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
        dual_channel_bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGR565)
        dual_channel_bgrf = dual_channel_bgr.astype('float32')/255.0

        # add depth as a third channel
        transformed_depth = transform_result['mask']
        transformed_depth = transformed_depth.astype('float32')

        img_out = np.zeros_like(img).astype('float32')
        img_out[:,:,0] = dual_channel_bgrf[:,:,0]
        img_out[:,:,1] = dual_channel_bgrf[:,:,1]
        img_out[:,:,2] = transformed_depth

        img_out = np.transpose(img_out, (2, 0, 1))
        mask = self.mask_generator(img_out, iter_i=self.iter_i)

        self.iter_i += 1
        return dict(image=img_out,
                    mask=mask)

class RandomChannelDrop(A.core.transforms_interface.ImageOnlyTransform):
    def __init__(self, always_apply=False, p=0.5):
        super(RandomChannelDrop, self).__init__(always_apply, p)

    def apply(self, image, **params):
        channel_to_drop = random.randint(0, 2)
        image[:, :, channel_to_drop] = 0
        return image
    

def get_transforms(transform_variant, out_size):
    if transform_variant == 'default':
        transform = A.Compose([
            A.RandomScale(scale_limit=0.2),  # +/- 20%
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.RandomCrop(height=out_size, width=out_size),
            A.HorizontalFlip(),
            A.CLAHE(),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.ToFloat()
        ])
    elif transform_variant == 'distortions':
        transform = A.Compose([
            IAAPerspective2(scale=(0.0, 0.06)),
            IAAAffine2(scale=(0.7, 1.3),
                       rotate=(-40, 40),
                       shear=(-0.1, 0.1)),
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.OpticalDistortion(),
            A.RandomCrop(height=out_size, width=out_size),
            A.HorizontalFlip(),
            A.CLAHE(),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.ToFloat()
        ])
    elif transform_variant == 'distortions_color_drop_05':
        transform = A.Compose([
            IAAPerspective2(scale=(0.0, 0.06)),
            IAAAffine2(scale=(0.7, 1.3),
                       rotate=(-40, 40),
                       shear=(-0.1, 0.1)),
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.OpticalDistortion(),
            A.RandomCrop(height=out_size, width=out_size),
            A.HorizontalFlip(),
            A.CLAHE(),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            RandomChannelDrop(p=0.5),
            A.ToFloat()
        ])
    elif transform_variant == 'distortions_color_drop_02':
        transform = A.Compose([
            IAAPerspective2(scale=(0.0, 0.06)),
            IAAAffine2(scale=(0.7, 1.3),
                       rotate=(-40, 40),
                       shear=(-0.1, 0.1)),
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.OpticalDistortion(),
            A.RandomCrop(height=out_size, width=out_size),
            A.HorizontalFlip(),
            A.CLAHE(),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            RandomChannelDrop(p=0.2),
            A.ToFloat()
        ])
    elif transform_variant == 'distortions_scale05_1':
        transform = A.Compose([
            IAAPerspective2(scale=(0.0, 0.06)),
            IAAAffine2(scale=(0.5, 1.0),
                       rotate=(-40, 40),
                       shear=(-0.1, 0.1),
                       p=1),
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.OpticalDistortion(),
            A.RandomCrop(height=out_size, width=out_size),
            A.HorizontalFlip(),
            A.CLAHE(),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.ToFloat()
        ])
    elif transform_variant == 'distortions_scale03_12':
        transform = A.Compose([
            IAAPerspective2(scale=(0.0, 0.06)),
            IAAAffine2(scale=(0.3, 1.2),
                       rotate=(-40, 40),
                       shear=(-0.1, 0.1),
                       p=1),
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.OpticalDistortion(),
            A.RandomCrop(height=out_size, width=out_size),
            A.HorizontalFlip(),
            A.CLAHE(),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.ToFloat()
        ])
    elif transform_variant == 'distortions_scale03_07':
        transform = A.Compose([
            IAAPerspective2(scale=(0.0, 0.06)),
            IAAAffine2(scale=(0.3, 0.7),  # scale 512 to 256 in average
                       rotate=(-40, 40),
                       shear=(-0.1, 0.1),
                       p=1),
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.OpticalDistortion(),
            A.RandomCrop(height=out_size, width=out_size),
            A.HorizontalFlip(),
            A.CLAHE(),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.ToFloat()
        ])
    elif transform_variant == 'distortions_light':
        transform = A.Compose([
            IAAPerspective2(scale=(0.0, 0.02)),
            IAAAffine2(scale=(0.8, 1.8),
                       rotate=(-20, 20),
                       shear=(-0.03, 0.03)),
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.RandomCrop(height=out_size, width=out_size),
            A.HorizontalFlip(),
            A.CLAHE(),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.ToFloat()
        ])
    elif transform_variant == 'non_space_transform':
        transform = A.Compose([
            A.CLAHE(),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.ToFloat()
        ])
    elif transform_variant == 'no_augs':
        transform = A.Compose([
            A.ToFloat()
        ])
    else:
        raise ValueError(f'Unexpected transform_variant {transform_variant}')
    return transform


def make_default_train_dataloader(indir, depth_datadir=None, hdf5_path=None, kind='default',
                                  out_size=512, mask_gen_kwargs=None, transform_variant='default',
                                  mask_generator_kind="mixed", dataloader_kwargs=None, ddp_kwargs=None, **kwargs):
    LOGGER.info(f'Make train dataloader {kind} from {indir}. Using mask generator={mask_generator_kind}')

    mask_generator = get_mask_generator(kind=mask_generator_kind, kwargs=mask_gen_kwargs)
    transform = get_transforms(transform_variant, out_size)

    if kind == 'default':
        dataset = InpaintingTrainDataset(indir=indir,
                                         mask_generator=mask_generator,
                                         transform=transform,
                                         **kwargs)
    elif kind == 'default_web':
        dataset = InpaintingTrainWebDataset(indir=indir,
                                            mask_generator=mask_generator,
                                            transform=transform,
                                            **kwargs)
    elif kind == 'img_with_segm':
        dataset = ImgSegmentationDataset(indir=indir,
                                         mask_generator=mask_generator,
                                         transform=transform,
                                         out_size=out_size,
                                         **kwargs)
    elif kind == 'img_with_depth_files' and depth_datadir is not None:
        dataset = DepthInpaintingTrainDataset(indir=indir,
                                         depth_datadir=depth_datadir,
                                         mask_generator=mask_generator,
                                         transform=transform,
                                         **kwargs)
    elif kind == 'img_with_depth_hdf5' and hdf5_path is not None:
        dataset = DepthInpaintingTrainWithHdf5Dataset(indir=indir,
                                         hdf5_path=hdf5_path,
                                         mask_generator=mask_generator,
                                         transform=transform,
                                         **kwargs)          
    elif kind == "rgb565d_with_depth_hdf5" and hdf5_path is not None:
        dataset = RGB565DInpaintingTrainWithHdf5Dataset(indir=indir,
                                         hdf5_path=hdf5_path,
                                         mask_generator=mask_generator,
                                         transform=transform,
                                         **kwargs)
    else:
        raise ValueError(f'Unknown train dataset kind {kind}')

    if dataloader_kwargs is None:
        dataloader_kwargs = {}

    is_dataset_only_iterable = kind in ('default_web',)

    if ddp_kwargs is not None and not is_dataset_only_iterable:
        dataloader_kwargs['shuffle'] = False
        dataloader_kwargs['sampler'] = DistributedSampler(dataset, **ddp_kwargs)

    if is_dataset_only_iterable and 'shuffle' in dataloader_kwargs:
        with open_dict(dataloader_kwargs):
            del dataloader_kwargs['shuffle']

    dataloader = DataLoader(dataset, **dataloader_kwargs)
    return dataloader


def make_default_val_dataset(indir, depth_datadir=None, hdf5_path=None, kind='default',
        out_size=512, transform_variant='default', **kwargs):
    if OmegaConf.is_list(indir) or isinstance(indir, (tuple, list)):
        return ConcatDataset([
            make_default_val_dataset(idir, kind=kind, out_size=out_size,
                transform_variant=transform_variant, **kwargs) for idir in indir 
        ])
    
    if depth_datadir is not None and (OmegaConf.is_list(depth_datadir) or isinstance(depth_datadir, (tuple, list))):
        return ConcatDataset([
            make_default_val_dataset(idir, depth_datadir=depth_datadir, kind=kind,
                out_size=out_size, transform_variant=transform_variant, **kwargs) for idir in indir 
        ])

    LOGGER.info(f'Make val dataloader {kind} from {indir}')
    mask_generator = get_mask_generator(kind=kwargs.get("mask_generator_kind"), kwargs=kwargs.get("mask_gen_kwargs"))

    if transform_variant is not None:
        transform = get_transforms(transform_variant, out_size)

    if kind == 'default':
        dataset = InpaintingEvaluationDataset(indir, **kwargs)
    elif kind == 'our_eval':
        dataset = OurInpaintingEvaluationDataset(indir, **kwargs)
    elif kind == 'img_with_segm':
        dataset = ImgSegmentationDataset(indir=indir,
                                         mask_generator=mask_generator,
                                         transform=transform,
                                         out_size=out_size,
                                         **kwargs)
    elif kind == 'online':
        dataset = InpaintingEvalOnlineDataset(indir=indir,
                                              mask_generator=mask_generator,
                                              transform=transform,
                                              out_size=out_size,
                                              **kwargs)
    elif kind == 'img_with_depth_files' and depth_datadir is not None:
        dataset = DepthInpaintingEvaluationDataset(indir, depth_datadir, **kwargs)
    elif kind == 'img_with_depth_hdf5' and hdf5_path is not None:
        dataset = DepthInpaintingEvaluationWithHdf5Dataset(indir, hdf5_path, **kwargs)
    elif kind == "rgb565d_with_depth_hdf5" and hdf5_path is not None:
        dataset = RGB565DInpaintingEvaluationWithHdf5Dataset(indir, hdf5_path, **kwargs)
    else:
        raise ValueError(f'Unknown val dataset kind {kind}')

    return dataset


def make_default_val_dataloader(*args, dataloader_kwargs=None, **kwargs):
    dataset = make_default_val_dataset(*args, **kwargs)

    if dataloader_kwargs is None:
        dataloader_kwargs = {}
    dataloader = DataLoader(dataset, **dataloader_kwargs)
    return dataloader


def make_constant_area_crop_params(img_height, img_width, min_size=128, max_size=512, area=256*256, round_to_mod=16):
    min_size = min(img_height, img_width, min_size)
    max_size = min(img_height, img_width, max_size)
    if random.random() < 0.5:
        out_height = min(max_size, ceil_modulo(random.randint(min_size, max_size), round_to_mod))
        out_width = min(max_size, ceil_modulo(area // out_height, round_to_mod))
    else:
        out_width = min(max_size, ceil_modulo(random.randint(min_size, max_size), round_to_mod))
        out_height = min(max_size, ceil_modulo(area // out_width, round_to_mod))

    start_y = random.randint(0, img_height - out_height)
    start_x = random.randint(0, img_width - out_width)
    return (start_y, start_x, out_height, out_width)
