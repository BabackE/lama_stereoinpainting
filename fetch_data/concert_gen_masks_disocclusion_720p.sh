TRAINING_ROOT=concert/train_720_disocclusion
DATASET_SUFFIX=disocclusion_720

python3 bin/gen_mask_dataset.py \
$(pwd)/configs/data_gen/disocclusion_$DATASET_SUFFIX.yaml \
$TRAINING_ROOT/val_source_$DATASET_SUFFIX/ \
$TRAINING_ROOT/val_$DATASET_SUFFIX/1_$DATASET_SUFFIX/

python3 bin/gen_mask_dataset.py \
$(pwd)/configs/data_gen/disocclusion_$DATASET_SUFFIX.yaml \
$TRAINING_ROOT/val_source_$DATASET_SUFFIX/ \
$TRAINING_ROOT/val_$DATASET_SUFFIX/2_$DATASET_SUFFIX/

python3 bin/gen_mask_dataset.py \
$(pwd)/configs/data_gen/disocclusion_$DATASET_SUFFIX.yaml \
$TRAINING_ROOT/val_source_$DATASET_SUFFIX/ \
$TRAINING_ROOT/val_$DATASET_SUFFIX/3_$DATASET_SUFFIX/

python3 bin/gen_mask_dataset.py \
$(pwd)/configs/data_gen/disocclusion_$DATASET_SUFFIX.yaml \
$TRAINING_ROOT/visual_test_source_$DATASET_SUFFIX/ \
$TRAINING_ROOT/visual_test_$DATASET_SUFFIX/1_$DATASET_SUFFIX/

python3 bin/gen_mask_dataset.py \
$(pwd)/configs/data_gen/disocclusion_$DATASET_SUFFIX.yaml \
$TRAINING_ROOT/visual_test_source_$DATASET_SUFFIX/ \
$TRAINING_ROOT/visual_test_$DATASET_SUFFIX/2_$DATASET_SUFFIX/

python3 bin/gen_mask_dataset.py \
$(pwd)/configs/data_gen/disocclusion_$DATASET_SUFFIX.yaml \
$TRAINING_ROOT/visual_test_source_$DATASET_SUFFIX/ \
$TRAINING_ROOT/visual_test_$DATASET_SUFFIX/3_$DATASET_SUFFIX/
