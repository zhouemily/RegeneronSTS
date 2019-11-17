set -x
#time python idt.py  dataset 8-4 tanh 50 8 
##time python idt.py  dataset 8-4 relu 50 8 

DATE=`date "+%Y%m%d"`
DATE_WITH_TIME=`date "+%Y%m%d-%H%M%S"` #add %3N as we want millisecond too

for act in relu sigmoid tanh
do
    for m in 100 200 500
    do
        for n in 8 16 32
        do
            time python idt.py dataset 64-32 $act $m $n 
        done
    done
done
