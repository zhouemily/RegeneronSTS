set -x
#time python idt.py  dataset 8-4 tanh 50 8 
##time python idt.py  dataset 8-4 relu 50 8 

DATE=`date "+%Y%m%d"`
DATE_WITH_TIME=`date "+%Y%m%d-%H%M%S"` #add %3N as we want millisecond too

for n in 1 2 3 
do
    for s in 64-32 64-32-16 64-32-16-8  
    do
        time python idt.py dataset $s relu 500 16 
    done
done

for n in 64 32 16 
do
    for m in 1 2 3
    do
        time python idtdeep.py  dataset 4 relu 500 16 $n
    done
done
