#!/bin/bash
# Copyright 2025 Keen Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# To run:
# chmod +x docker_run.sh
#
# Both the code folder and image_name are optional, if not provided the code folder will default to the current directory.
#./docker_run.sh /path/to/code
#./docker_run.sh /path/to/code custom_image_name
#
# Inside the container, /path/to/code will be mounted as /workspaces/code

DEFAULT_IMAGE_NAME="keen_physical_gpu"

CODE_FOLDER_PATH=${1:-$(pwd)}
DOCKER_IMAGE_NAME=${2:-$DEFAULT_IMAGE_NAME}

# Get the last directory in the path for the mount point (basename of the code folder)
MOUNT_POINT="/workspaces/$(basename "$CODE_FOLDER_PATH")"

# Run the docker container with the specified image name and mount the code folder to the container
docker run --rm -it \
  -v "$CODE_FOLDER_PATH:$MOUNT_POINT" \
  -w "$MOUNT_POINT" \
  --mount source=/dev,target=/dev,type=bind \
  -e DISPLAY=$DISPLAY \
  -e XAUTHORITY=/root/.Xauthority \
  --gpus=all \
  --privileged \
  --network=host \
  --ipc=host \
  --ulimit=memlock=-1 \
  --ulimit=stack=67108864 \
  --cap-add=SYS_PTRACE \
  --volume=/tmp/.X11-unix:/tmp/.X11-unix \
  -v /home/adam/.Xauthority:/root/.Xauthority \
  "$DOCKER_IMAGE_NAME"
