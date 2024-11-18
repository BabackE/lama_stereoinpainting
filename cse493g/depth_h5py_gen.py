import sys
sys.path.append("../../uw/phd_repo")

import h5py
import os
import cv2
import numpy as np
from tqdm.notebook import tqdm

def refresh_depth_estimator():
    from depthfactory.depthanything import SimpleDepthAnythingEvaluator2
    import torch
    import gc

    print("Refreshing depth estimator")

    torch.cuda.empty_cache()
    gc.collect()

    depth_anything = SimpleDepthAnythingEvaluator2()
    return depth_anything



def create_hdf5_from_directory(root_dir, hdf5_file_path, refresh_threshold=5000):
    process_count = 0
    depth_anything = None
    with h5py.File(hdf5_file_path, 'a') as hdf5_file:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            skipped_count = 0
            # Compute the relative path and use it to create groups
            relative_path = os.path.relpath(dirpath, root_dir)
            group_path = relative_path.replace(os.sep, '/')
            # Create groups recursively
            group = hdf5_file.require_group(group_path)
           
            for filename in filenames:
                dataset_name = os.path.splitext(filename)[0]
                full_dataset_path = group_path + '/' + dataset_name

                # Check if dataset already exists in the HDF5 file
                if full_dataset_path in hdf5_file:
                    skipped_count += 1
                    continue
            
            if skipped_count < len(filenames):
                print(f"Processing {group_path} with {len(filenames)-skipped_count} files.")
                with tqdm(total=len(filenames), desc=f"Depth gen", unit="item") as pbar:
                    for filename in tqdm(filenames, desc=group_path):
                        dataset_name = os.path.splitext(filename)[0]
                        full_dataset_path = group_path + '/' + dataset_name
                        pbar.set_description(f"{full_dataset_path}")

                        # Check if dataset already exists in the HDF5 file
                        if full_dataset_path in hdf5_file:
                            skipped_count += 1
                            continue

                        if process_count % refresh_threshold == 0:
                            if depth_anything is not None:
                                del depth_anything
                            depth_anything = refresh_depth_estimator()
                            
                        # Assuming files are depth maps stored as NumPy arrays
                        file_path = os.path.join(dirpath, filename)
                        img = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
                        depth_map = depth_anything.get_rawdepth(img)
                        
                        # normalize depth map, subtract mean and divide by std
                        depth_map_min = np.min(depth_map)
                        depth_map_range = np.max(depth_map) - depth_map_min
                        depth_map = (depth_map - depth_map_min) / depth_map_range
                        depth_map_u16 = (depth_map * 65535).astype(np.uint16)

                        # Save the depth map in the group
                        dataset = group.create_dataset(dataset_name, data=depth_map_u16, compression="gzip")
                        dataset.attrs['min'] = depth_map_min
                        dataset.attrs['range'] = depth_map_range

                        process_count+=1
                        pbar.update(1)
            
                print(f"Skipped {skipped_count} files in {group_path}.")


def list_all_groups_and_datasets(hdf5_file):
    for name, obj in hdf5_file.items():
        if isinstance(obj, h5py.Group):
            print(f"Group: {obj.name}")
            list_all_groups_and_datasets(obj)
        elif isinstance(obj, h5py.Dataset):
            print(f"  Dataset: {obj.name}")




def access_depth_map(hdf5_file_path, dataset_path):
    """
    Accesses and reconstructs a depth map from the HDF5 file using stored mean and std attributes.

    Parameters:
        hdf5_file_path (str): Path to the HDF5 file.
        dataset_path (str): Path to the dataset within the HDF5 file.

    Returns:
        numpy.ndarray: The reconstructed original depth map, or None if the dataset does not exist.
    """
    print(f"Accessing depth map at '{dataset_path}' in {hdf5_file_path}...")
    with h5py.File(hdf5_file_path, 'r') as hdf5_file:
        if dataset_path in hdf5_file:
            # Access the dataset
            dataset = hdf5_file[dataset_path]
            depth_map_u16 = dataset[:]
            
            # Retrieve mean and std attributes
            depth_map_min = dataset.attrs['min']
            depth_map_range = dataset.attrs['range']
            
            # Convert uint16 back to normalized float
            divisor = 65535.0 if depth_map_u16.dtype == np.uint16 else 255.0 
            depth_map_normalized = depth_map_u16.astype(np.float32) / divisor
            
            # Denormalize to reconstruct the original depth map
            depth_map_reconstructed = depth_map_normalized * depth_map_range + depth_map_min
            print(depth_map_reconstructed.min(), depth_map_reconstructed.max())
            
            print(f"Accessed and reconstructed depth map at '{dataset_path}' with shape {depth_map_reconstructed.shape}")
            return depth_map_reconstructed
        else:
            print(f"Depth map '{dataset_path}' does not exist.")
            return None

# updates a specific dataset in the hdf5 file with a new depth map
def update_dataset_in_hdf5(hdf5_file_path, dataset_path, new_depth_map):
    with h5py.File(hdf5_file_path, 'a') as hdf5_file:
        if dataset_path in hdf5_file:
            # Normalize depth map, subtract mean and divide by std
            depth_map_min = np.min(new_depth_map)
            depth_map_range = np.max(new_depth_map) - depth_map_min
            depth_map = (new_depth_map - depth_map_min) / depth_map_range
            depth_map_u16 = (depth_map * 65535).astype(np.uint16)

            # Update existing dataset
            hdf5_file[dataset_path][...] = depth_map_u16

            # Update attributes
            hdf5_file[dataset_path].attrs['min'] = depth_map_min
            hdf5_file[dataset_path].attrs['range'] = depth_map_range

            print(f"Updated dataset '{dataset_path}' successfully.")
        else:
            print(f"Dataset '{dataset_path}' does not exist in the HDF5 file.")

# takes a u16 hdf5 file and converts it to u8
def copy_hdf5_to_u8(source_hdf5_path, target_hdf5_path):
    with h5py.File(source_hdf5_path, 'r') as source_file, h5py.File(target_hdf5_path, 'a') as target_file:
        def recursive_copy(group_source, group_target, group_path="/"):
            keys = list(group_source.keys())
            with tqdm(total=len(keys), desc=f"Copying groups/datasets", unit="item") as pbar:
                for key in group_source:
                    obj = group_source[key]
                    current_path = f"{group_path}{key}"
                    pbar.set_description(f"{current_path}")

                    
                    if isinstance(obj, h5py.Group):
                        # Check if group already exists in target file
                        if key not in group_target:
                            # Create the group if it does not exist
                            new_group = group_target.create_group(key)
                        else:
                            new_group = group_target[key]
                        # Recursively copy the contents of the group
                        recursive_copy(obj, new_group, current_path + "/")
                    elif isinstance(obj, h5py.Dataset):
                        if key in group_target:
                            pbar.update(1)
                            continue
                        
                        # Read the u16 dataset and convert to u8
                        depth_map_u16 = obj[:]
                        
                        # Retrieve mean and std attributes
                        depth_map_min = obj.attrs['min']
                        depth_map_range = obj.attrs['range']
                        
                        # Convert uint16 back to normalized float
                        depth_map_normalized = depth_map_u16.astype(np.float32) / 65535.0
                        
                        # Denormalize to reconstruct the original depth map
                        depth_map_reconstructed = depth_map_normalized * depth_map_range + depth_map_min

                        # convert to u8
                        reconstructed_min = np.min(depth_map_reconstructed)
                        reconstructed_range = np.max(depth_map_reconstructed) - reconstructed_min
                        depth_map_u8 = (depth_map_reconstructed - reconstructed_min) / reconstructed_range
                        depth_map_u8 = (depth_map_u8 * 255).astype(np.uint8)

                        # Save the depth map in the group
                        u8_dataset = group_target.create_dataset(key, data=depth_map_u8, dtype='uint8', compression="gzip", compression_opts=9)
                        u8_dataset.attrs['min'] = reconstructed_min
                        u8_dataset.attrs['range'] = reconstructed_range

                    pbar.update(1)

        # Start the recursive copying
        recursive_copy(source_file, target_file)

def hist_plot(depth, title='Histogram of Depth Values', xlabel='Depth'):
    import matplotlib.pyplot as plt
    print(depth.min(), depth.max())
    flattened_depth = depth.flatten()
    num_bins = int(np.sqrt(len(flattened_depth)))  # A common choice is the square root of the number of data points

    plt.figure(figsize=(10, 6))
    plt.hist(flattened_depth, bins=num_bins, color='blue', alpha=0.7)  # Adjust color and transparency as needed
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='List files in hdf5 file.')
    parser.add_argument('--hdf5_file', type=str, help='Root directory containing images.')
    args = parser.parse_args()

    with h5py.File(args.hdf5_file, 'r') as hdf5_file:
        list_all_groups_and_datasets(hdf5_file)
