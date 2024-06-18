# Split: split train -> train & val
echo creating random set
cat concert/fetch_lists/train_shuffled.flist | shuf > concert/train256/temp_train_shuffled.flist
cat concert/train256/temp_train_shuffled.flist | head -n 2000 > concert/train256/val_shuffled.flist
cat concert/train256/temp_train_shuffled.flist | tail -n +2001 > concert/train256/train_shuffled.flist
cat concert/fetch_lists/val_shuffled.flist > concert/train256/visual_test_shuffled.flist

mkdir concert/train256/train_256/
mkdir concert/train256/val_source_256/
mkdir concert/train256/visual_test_source_256/

echo moving images into assigned buckets
cat concert/train256/train_shuffled.flist | xargs -I {} mv concert/train256/1080p/{} concert/train256/train_256/
cat concert/train256/val_shuffled.flist | xargs -I {} mv concert/train256/1080p/{} concert/train256/val_source_256/
cat concert/train256/visual_test_shuffled.flist | xargs -I {} mv concert/train256/1080p/{} concert/train256/visual_test_source_256/

echo creating training scripts

# create location config concert.yaml
PWD=$(pwd)
DATASET=${PWD}/concert/train256
CONCERT_DST=${PWD}/configs/training/location/concert256.yaml
CONCERT_SRC=${PWD}/concert/concert256.yaml

touch $CONCERT_SRC
echo "# @package _group_" >> $CONCERT_SRC
echo "data_root_dir: ${DATASET}/" >> $CONCERT_SRC
echo "out_root_dir: ${PWD}/experiments/" >> $CONCERT_SRC
echo "tb_dir: ${PWD}/outputs/tb_logs/" >> $CONCERT_SRC
echo "pretrained_models: ${PWD}/" >> $CONCERT_SRC

sudo cp $CONCERT_SRC $CONCERT_DST