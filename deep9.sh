#!/bin/bash
#set -x
export TIMEFORMAT=%R
#for m in 16
for m in 16 32 64
do
    #for i in 1
    for i in 1 2 3 4 5 6 7 8 9
    do
        (time python idtdeep.py  dataset $i relu 200 16 $m) 2>time.data 
        mytime=`tail -1 time.data`
        echo "mytime=$mytime"
    done
done
