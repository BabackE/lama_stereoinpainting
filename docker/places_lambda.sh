#!/bin/bash

# Set the current and source directories
CURDIR=$(pwd)
SRCDIR=$(dirname "$CURDIR")

# Run the Docker container with the specified options
docker run --gpus all -it --rm --shm-size=8gb --env="DISPLAY" \
--volume="${SRCDIR}/../ade20k:/home/user/lama/ade20k" \
--volume="${SRCDIR}/../experiments:/home/user/lama/experiments" \
--volume="${SRCDIR}/../outputs:/home/user/lama/outputs" \
--volume="${SRCDIR}/../outputs/tb_logs:/home/user/lama/tb_logs" \
--volume="${SRCDIR}/../places/depth:/home/user/lama/places_depth" \
--volume="${SRCDIR}/../places/depth:/home/user/lama/places_standard_dataset" \
--volume="${SRCDIR}/docker/.ssh:/home/user/.ssh" \
--name="lama" lama /bin/bash
