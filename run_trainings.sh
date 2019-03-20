#!/bin/sh

#run this script to run all training

for i in train_*.py; do echo "starting $i"; python $i; done