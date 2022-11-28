#!/bin/bash
sudo docker run --gpus all -v $(pwd):/workspace --ipc=host --rm  -p 8888:8888  pytorch_env jupyter notebook --allow-root --ip 0.0.0.0