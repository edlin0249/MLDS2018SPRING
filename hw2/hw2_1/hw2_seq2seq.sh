#!/usr/bin/env bash

# $1: testing data directory
# $2: output filename
# $3: dictionary path
# $4: model path

python3 test.py $1 $2 "./lang_train" "./best_model"
