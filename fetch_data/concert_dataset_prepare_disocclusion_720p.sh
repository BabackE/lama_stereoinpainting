TRAINING_ROOT=concert/train_720_disocclusion
DATASET_SUFFIX=disocclusion_720

# Split: split train -> train & val
echo creating random set
cat $TRAINING_ROOT/fetch_lists/train_shuffled.flist | shuf > $TRAINING_ROOT/temp_train_shuffled.flist
cat $TRAINING_ROOT/temp_train_shuffled.flist | head -n 2000 > $TRAINING_ROOT/val_shuffled.flist
cat $TRAINING_ROOT/temp_train_shuffled.flist | tail -n +2001 > $TRAINING_ROOT/train_shuffled.flist
cat $TRAINING_ROOT/fetch_lists/val_shuffled.flist > $TRAINING_ROOT/visual_test_shuffled.flist

mkdir $TRAINING_ROOT/train_$DATASET_SUFFIX/
mkdir $TRAINING_ROOT/val_source_$DATASET_SUFFIX/
mkdir $TRAINING_ROOT/visual_test_source_$DATASET_SUFFIX/

echo moving images into assigned buckets
echo training set
cat $TRAINING_ROOT/train_shuffled.flist | xargs -I {} mv $TRAINING_ROOT/$DATASET_SUFFIX/{} $TRAINING_ROOT/train_$DATASET_SUFFIX/
echo val set
cat $TRAINING_ROOT/val_shuffled.flist | xargs -I {} mv $TRAINING_ROOT/$DATASET_SUFFIX/{} $TRAINING_ROOT/val_source_$DATASET_SUFFIX/
echo visual test set
cat $TRAINING_ROOT/visual_test_shuffled.flist | xargs -I {} mv $TRAINING_ROOT/$DATASET_SUFFIX/{} $TRAINING_ROOT/visual_test_source_$DATASET_SUFFIX/

echo creating training scripts

# create location config concert.yaml
PWD=$(pwd)
DATASET=${PWD}/$TRAINING_ROOT
CONCERT_DST=${PWD}/configs/training/location/concert_{$DATASET_SUFFIX}_rand.yaml
CONCERT_SRC=${PWD}/concert/concert_{$DATASET_SUFFIX}_rand.yaml

touch $CONCERT_SRC
echo "# @package _group_" >> $CONCERT_SRC
echo "data_root_dir: ${DATASET}/" >> $CONCERT_SRC
echo "out_root_dir: ${PWD}/experiments/" >> $CONCERT_SRC
echo "tb_dir: ${PWD}/outputs/tb_logs/" >> $CONCERT_SRC
echo "pretrained_models: ${PWD}/" >> $CONCERT_SRC

sudo cp $CONCERT_SRC $CONCERT_DST