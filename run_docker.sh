USER=$(whoami)
docker run \
    -it \
    --gpus all \
    --ipc=host \
    -v='/data1/':/data1 \
    -v='/data2/':/data2 \
    -v='/data3/':/data3 \
    -v='/data4/':/data4 \
	-p=60800-60809:60800-60809\
    --name='kma_2024'\
    --shm-size=24G \
    kma_2024:latest /bin/bash
