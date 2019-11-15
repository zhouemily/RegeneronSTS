set -x
export TIMEFORMAT=%R
for m in 50 100 200 500
do
   for n in 4 8 16 32
   do
      time python idt.py  dataset 8-4 relu $m $n 
      time python idt.py  dataset 64-16 relu $m $n 
      time python idt.py  dataset 128-32 relu $m $n 
   done
done
