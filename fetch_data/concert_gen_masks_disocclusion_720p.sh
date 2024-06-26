TRAINING_ROOT=concert/train_720_disocclusion
DATASET_SUFFIX=disocclusion_720

echo val_source/1_$DATASET_SUFFIX
python3 bin/gen_mask_dataset.py \
$(pwd)/configs/data_gen/$DATASET_SUFFIX.yaml \
$TRAINING_ROOT/val_source_$DATASET_SUFFIX/ \
$TRAINING_ROOT/val_$DATASET_SUFFIX/1_$DATASET_SUFFIX/

echo val_source/2_$DATASET_SUFFIX
python3 bin/gen_mask_dataset.py \
$(pwd)/configs/data_gen/$DATASET_SUFFIX.yaml \
$TRAINING_ROOT/val_source_$DATASET_SUFFIX/ \
$TRAINING_ROOT/val_$DATASET_SUFFIX/2_$DATASET_SUFFIX/

echo val_source/3_$DATASET_SUFFIX
python3 bin/gen_mask_dataset.py \
$(pwd)/configs/data_gen/$DATASET_SUFFIX.yaml \
$TRAINING_ROOT/val_source_$DATASET_SUFFIX/ \
$TRAINING_ROOT/val_$DATASET_SUFFIX/3_$DATASET_SUFFIX/

echo visual_test/1_$DATASET_SUFFIX
python3 bin/gen_mask_dataset.py \
$(pwd)/configs/data_gen/$DATASET_SUFFIX.yaml \
$TRAINING_ROOT/visual_test_source_$DATASET_SUFFIX/ \
$TRAINING_ROOT/visual_test_$DATASET_SUFFIX/1_$DATASET_SUFFIX/
