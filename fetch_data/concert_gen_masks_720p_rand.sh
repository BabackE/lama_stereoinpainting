TRAINING_ROOT=concert/train_720_rand
DATASET_SUFFIX=720

python3 bin/gen_mask_dataset.py \
$(pwd)/configs/data_gen/random_thick_$DATASET_SUFFIX.yaml \
$TRAINING_ROOT/val_source_$DATASET_SUFFIX/ \
$TRAINING_ROOT/val_$DATASET_SUFFIX/random_thick_$DATASET_SUFFIX/

python3 bin/gen_mask_dataset.py \
$(pwd)/configs/data_gen/random_thin_$DATASET_SUFFIX.yaml \
$TRAINING_ROOT/val_source_$DATASET_SUFFIX/ \
$TRAINING_ROOT/val_$DATASET_SUFFIX/random_thin_$DATASET_SUFFIX/

python3 bin/gen_mask_dataset.py \
$(pwd)/configs/data_gen/random_medium_$DATASET_SUFFIX.yaml \
$TRAINING_ROOT/val_source_$DATASET_SUFFIX/ \
$TRAINING_ROOT/val_$DATASET_SUFFIX/random_medium_$DATASET_SUFFIX/

python3 bin/gen_mask_dataset.py \
$(pwd)/configs/data_gen/random_thick_$DATASET_SUFFIX.yaml \
$TRAINING_ROOT/visual_test_source_$DATASET_SUFFIX/ \
$TRAINING_ROOT/visual_test_$DATASET_SUFFIX/random_thick_$DATASET_SUFFIX/

python3 bin/gen_mask_dataset.py \
$(pwd)/configs/data_gen/random_thin_$DATASET_SUFFIX.yaml \
$TRAINING_ROOT/visual_test_source_$DATASET_SUFFIX/ \
$TRAINING_ROOT/visual_test_$DATASET_SUFFIX/random_thin_$DATASET_SUFFIX/

python3 bin/gen_mask_dataset.py \
$(pwd)/configs/data_gen/random_medium_$DATASET_SUFFIX.yaml \
$TRAINING_ROOT/visual_test_source_$DATASET_SUFFIX/ \
$TRAINING_ROOT/visual_test_$DATASET_SUFFIX/random_medium_$DATASET_SUFFIX/
