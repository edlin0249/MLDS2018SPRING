#!/usr/bin/env bash

python gan_test.py generate --netd_path netd --netg_path netg
rm -rf __pycache__