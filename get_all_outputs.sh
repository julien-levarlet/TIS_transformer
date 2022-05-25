#!/bin/bash

# logs directory should be :
# lightning_logs 
#   - [name]_chrx[_info]
#       - version0
#           - checkpoints/[name].ckpt
#       - version1
#           - checkpoints/[name].ckpt
#

if [ "$#" -ne 3 ]; then
    echo "Illegal number of parameters, need pytorch lightning save directory, TIS_transformer absolute path and out parent directory"
    exit 1
fi

TIS_transformer_path=$2

for model in $TIS_transformer_path/$1*;
do
    # get model name
    folder=${model#/}
    exp_name=${model##*/}
    sub="chr"
    if [[ "$exp_name" == *"$sub"* ]];
    then
        # get chromosome name
        chr=${exp_name#*_}
        chr=${chr%%_*}
        
        for folder in $model/*;
        do
            # get version and path to the model
            version=${folder##*/}            
            path="$folder/checkpoints/*"
            save_path="$3/out/${exp_name}_${version}.npy"
            for checkpoint in $path; # done only once
            do
                # impute on the model 
                echo "model path : $checkpoint"
                python3 $TIS_transformer_path/TIS_transformer.py impute "$TIS_transformer_path/data/GRCh38p13_unzip/$chr.npy" $checkpoint --gpu 1 --save_path $save_path --num_workers 3
            done
        done
    else
        echo "$exp_name does not contain a chromome name"
    fi
done

