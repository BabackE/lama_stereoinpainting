#!/bin/sh
sudo rm /home/user/places_standard_dataset/train_large_places365standard_disparity_u8.hdf5
sudo rm /home/user/places_standard_dataset/val_large_places365standard_disparity_u8.hdf5
sudo ln -s /home/user/places_standard_dataset/depth/train_large_places365standard_disparity_u8.hdf5 /home/user/places_standard_dataset/train_large_places365standard_disparity_u8.hdf5 
sudo ln -s /home/user/places_standard_dataset/depth/val_large_places365standard_disparity_u8.hdf5 /home/user/places_standard_dataset/val_large_places365standard_disparity_u8.hdf5 
