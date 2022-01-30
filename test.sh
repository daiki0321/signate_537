#!/bin/bash

set -e

cd ./output/$1 && ./yolo_main  > log.txt

grep "nboxes = 5" log.txt
