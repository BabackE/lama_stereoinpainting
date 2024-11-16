SET CURDIR=%CD%
SET SRCDIR=%CD%\..

docker run --gpus all -it --rm --shm-size=8gb --env="DISPLAY" ^
--volume="%SRCDIR%\ade20k":/home/user/lama/ade20k ^
--volume="%SRCDIR%\celeba-hq-dataset":/home/user/lama/celeba-hq-dataset ^
--volume="D:\experiments\stereo\e113-lama-training-datasets\se03_bono_720p_stereo_mask_training\720_rand_dataset\concert":/home/user/lama/concert ^
--volume="%SRCDIR%\experiments":/home/user/lama/experiments ^
--volume="%SRCDIR%\docker\places\places_standard_dataset":/home/user/lama/places_standard_dataset ^
--volume="%SRCDIR%\outputs":/home/user/lama/outputs ^
--volume="%SRCDIR%\outputs\tb_logs":/home/user/lama/tb_logs ^
--volume="%SRCDIR%\models\models":/home/user/lama/models/models ^
--volume="%SRCDIR%\docker\.ssh\id_ed25119":/home/user/.ssh/id_ed25119 ^
--volume="%SRCDIR%\hub":/home/user/lama/hub ^
--name="lama" lama /bin/bash
