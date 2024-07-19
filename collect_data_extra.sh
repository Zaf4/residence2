#!/bin/bash

mkdir -p $HOME/data_extra

KTS=("2.80" "3.00" "3.50" "4.00")
UMS=("10" "20" "40" "60")

for KT in ${KTS[@]}; do
    for UM in ${UMS[@]}; do
        PATTERN=/home/gottar/5x10t/${KT}/${UM}/data.extra
            echo $PATTERN
            cp $PATTERN $HOME/data_extra/${KT}_${UM}.extra
    done
done

cd $HOME
tar -czvf data_extra.tar.gz data_extra/