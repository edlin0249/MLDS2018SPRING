#!/usr/bin/env bash

python3 gan_test.py generate --netd_path netd --netg_path netg
rm -rf __pycache__