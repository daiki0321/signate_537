#!/bin/bash

set -e

cd ./output/$1 && ./yolo_main ./image_list_test.txt > log.txt

grep "nboxes = 5" log.txt
