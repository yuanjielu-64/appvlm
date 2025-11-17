#!/bin/bash

killall -9 rosmaster

killall gzclient

killall gzserver

for i in {0..49} ; do
    n=`expr $i \* 6` # 50 test BARN worlds with equal spacing indices: [0, 6, 12, ..., 294]
    for j in {1..10} ; do 
        python3 run_ddp.py --world_idx $n --out "out_test_ddp_p=1.4.txt"
        sleep 4
    done
done