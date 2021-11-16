#!/bin/bash

MAIN=/data/home/rndkoa/2021
CONFIG=${MAIN}/SRGAN/DABA/config.json
SRCS=${MAIN}/SRGAN/SRCS
PYTHON=${MAIN}/envs/kma_deep_env/bin/python3
export PYTHONPATH=${PYTHON}:${SRCS}
export ECCODES_DEFINITION_PATH=/data/home/rndkoa/2021/envs/kma_deep_env/share/eccodes/definitions

${PYTHON} ${SRCS}/main.py --config ${CONFIG}
