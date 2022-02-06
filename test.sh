#!/bin/bash

set -e

export LD_LIBRARY_PATH=/usr/local/lib

cd ./output/$1 && ./yolo_main ./image_list_test.txt > log.txt

grep "ID = 8 class = car " log.txt
grep "ID = 12 class = person " log.txt
