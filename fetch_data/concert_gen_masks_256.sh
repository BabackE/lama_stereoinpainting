python3 bin/gen_mask_dataset.py \
$(pwd)/configs/data_gen/random_thick_256.yaml \
concert/train/val_source_256/ \
concert/train/val_256/random_thick_256/

python3 bin/gen_mask_dataset.py \
$(pwd)/configs/data_gen/random_thin_256.yaml \
concert/train/val_source_256/ \
concert/train/val_256/random_thin_256/

python3 bin/gen_mask_dataset.py \
$(pwd)/configs/data_gen/random_medium_256.yaml \
concert/train/val_source_256/ \
concert/train/val_256/random_medium_256/

python3 bin/gen_mask_dataset.py \
$(pwd)/configs/data_gen/random_thick_256.yaml \
concert/train/visual_test_source_256/ \
concert/train/visual_test_256/random_thick_256/

python3 bin/gen_mask_dataset.py \
$(pwd)/configs/data_gen/random_thin_256.yaml \
concert/train/visual_test_source_256/ \
concert/train/visual_test_256/random_thin_256/

python3 bin/gen_mask_dataset.py \
$(pwd)/configs/data_gen/random_medium_256.yaml \
concert/train/visual_test_source_256/ \
concert/train/visual_test_256/random_medium_256/
