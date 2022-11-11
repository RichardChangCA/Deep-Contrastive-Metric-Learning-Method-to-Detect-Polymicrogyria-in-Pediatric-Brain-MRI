#!/bin/bash

# "9 22 12 13 23 34 6 10 18 14 16 30 20 8 17 31 29 3 28 4 2 5 19"

#fold 0
test_list="13 23 34 6"

#inner fold 0
val_list="4 2 5 19"

python3 metric_learning_classification.py --test="13 23 34 6" --val="4 2 5 19" --inner_fold=0

#inner fold 1
val_list="31 29 3 28"

python3 metric_learning_classification.py --test="13 23 34 6" --val="31 29 3 28" --inner_fold=1

#inner fold 2
val_list="30 20 8 17"

python3 metric_learning_classification.py --test="13 23 34 6" --val="30 20 8 17" --inner_fold=2

#inner fold 3
val_list="10 18 14 16"

python3 metric_learning_classification.py --test="13 23 34 6" --val="10 18 14 16" --inner_fold=3
