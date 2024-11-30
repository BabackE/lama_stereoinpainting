SET CURDIR=%CD%
SET SRCDIR=%CD%\..

docker run --gpus all -it --rm --shm-size=8gb --env="DISPLAY" ^
--volume="%SRCDIR%\ade20k":/home/user/lama/ade20k ^
--volume="%SRCDIR%\experiments":/home/user/lama/experiments ^
--volume="%SRCDIR%\outputs":/home/user/lama/outputs ^
--volume="%SRCDIR%\outputs\tb_logs":/home/user/lama/tb_logs ^
--volume="%SRCDIR%\models\models":/home/user/lama/models/models ^
--volume="%SRCDIR%\hub":/home/user/lama/hub ^
--volume="%SRCDIR%\docker\places\depth":/home/user/lama/places_depth ^
--volume="%SRCDIR%\docker\.ssh":/home/user/.ssh ^
--name=%1 lama_places_depth /bin/bash
