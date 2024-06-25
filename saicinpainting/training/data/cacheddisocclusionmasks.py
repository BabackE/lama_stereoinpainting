import logging
from enum import Enum

import cv2
import numpy as np
import os
import sys

LOGGER = logging.getLogger(__name__)

from saicinpainting.training.data.masks import MixedMaskGenerator

class CachedDisocclusionMaskGenerator:

    def __init__(self, depth_cut_options=True, depth_cutoff=0.23, seed=-1, combine_with_random_masks=True):
        self.depth_cut_options = depth_cut_options
        self.depth_cutoff = depth_cutoff
        self.combine_with_random_masks = combine_with_random_masks
        if (seed != -1):
            np.random.seed(seed)

    def get_resource_path_from_img_path(self, path, resource_path, resource_suffix):
        # Split the original path to get directory and filename
        dir_name, file_name = os.path.split(path)
        
        # Remove the last directory (train_set) and add resource path (e.g. occlusions)
        new_dir_name = os.path.join(os.path.dirname(dir_name), resource_path)
        
        # Change the file extension and append resource suffix (e.g. '_rawdepth.png')
        new_file_name = os.path.splitext(file_name)[0] + resource_suffix
        
        # Construct the new path
        new_path = os.path.join(new_dir_name, new_file_name)

        if (not os.path.exists(new_path)):
            LOGGER.error(f"Resource Not Found: {new_path}")
            sys.exit(-1)
        
        return new_path


    class DisocclusionOptions(Enum):
        LEFT_DISOCCLUSION = 0
        RIGHT_DISOCCLUSION = 1
        COMBINED_DISOCCLUSION =  2

    class DepthCutOptions(Enum):
        FOREGROUND = 0
        BACKGROUND = 1
        NONE = 2

    default_irregular_kwargs = {
        "max_angle": 4,
        "max_len": 200,
        "max_width": 500,
        "max_times": 4,
        "min_times": 1,
    }

    default_box_kwargs = {
        "margin": 10,
        "bbox_min_size": 84,
        "bbox_max_size": 421,
        "max_times": 6,
        "min_times": 1 
    }

    def create_selected_mask(self, img_path, selected_disocclusion_type, selected_depthcutoff_type, iter_i=-1):
        LOGGER.info(f"[DisocclusionMask {iter_i}]: \ndisocclusion_type = {selected_disocclusion_type}\n depth_cutoff_type = {selected_depthcutoff_type}\n random_masks= {self.combine_with_random_masks}")

        # load the type of mask we want to use
        selected_disocclusion_mask = 0

        left_disocclusion_path = self.get_resource_path_from_img_path(img_path, "inpaint_left", "_left_dilated.png")
        right_disocclusion_path = self.get_resource_path_from_img_path(img_path, "inpaint_right", "_right_dilated.png")
        if (selected_disocclusion_type == self.DisocclusionOptions.COMBINED_DISOCCLUSION):
            left_disocclusion = cv2.imread(left_disocclusion_path, cv2.IMREAD_GRAYSCALE)
            right_disocclusion = cv2.imread(right_disocclusion_path, cv2.IMREAD_GRAYSCALE)
            selected_disocclusion_mask = cv2.bitwise_or(left_disocclusion, right_disocclusion)
        elif (selected_disocclusion_type == self.DisocclusionOptions.LEFT_DISOCCLUSION):
            selected_disocclusion_mask = cv2.imread(left_disocclusion_path, cv2.IMREAD_GRAYSCALE)
        else:
            selected_disocclusion_mask = cv2.imread(right_disocclusion_path, cv2.IMREAD_GRAYSCALE)

        # select foreground, background or both parts of the disocclusion
        if (self.depth_cut_options and selected_depthcutoff_type != self.DepthCutOptions.NONE):
            raw_depth_path = self.get_resource_path_from_img_path(img_path, "rawdepth", "_rawdepth.png")
            raw_depth = cv2.imread(raw_depth_path, -1)
            depth_cutoff = raw_depth.max() * self.depth_cutoff
            if (selected_depthcutoff_type == self.DepthCutOptions.FOREGROUND):
                foreground_cutoff = (raw_depth > depth_cutoff).astype(np.uint8)
                selected_disocclusion_mask = cv2.bitwise_and(foreground_cutoff, selected_disocclusion_mask).astype(np.uint8) * 255
            elif (selected_depthcutoff_type == self.DepthCutOptions.BACKGROUND):
                background_cutoff = (raw_depth <= depth_cutoff).astype(np.uint8)
                selected_disocclusion_mask = cv2.bitwise_and(background_cutoff, selected_disocclusion_mask).astype(np.uint8) * 255

        selected_disocclusion_mask = selected_disocclusion_mask.astype(np.float32)/255.0
        selected_disocclusion_mask = np.expand_dims(selected_disocclusion_mask, axis=0)

        if (self.combine_with_random_masks):
            mask_generator = MixedMaskGenerator(box_proba=1, segm_proba=0, irregular_proba=1, irregular_kwargs=self.default_irregular_kwargs, box_kwargs=self.default_box_kwargs)
            random_masks = mask_generator(selected_disocclusion_mask)
            selected_disocclusion_mask = cv2.bitwise_or(selected_disocclusion_mask, random_masks)

        return selected_disocclusion_mask


    def __call__(self, img_path, iter_i = None, raw_image=None):
        # pick the type of disocclusion
        selected_disocclusion_type = np.random.choice(list(CachedDisocclusionMaskGenerator.DisocclusionOptions), p=[0.8,0.15,0.05])

        # pick what type depth cutoff we'll have
        selected_depthcutoff_type = np.random.choice(list(CachedDisocclusionMaskGenerator.DepthCutOptions), p=[0.125,0.125,0.75])

        return self.create_selected_mask(img_path, selected_disocclusion_type, selected_depthcutoff_type, iter_i=iter_i)
