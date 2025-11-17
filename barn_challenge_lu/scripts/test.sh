#!/bin/bash

killall -9 rosmaster

killall gzclient

killall gzserver

for i in {300..359} ; do
    for j in {1..3} ; do            
        python run_ddp.py --world_idx $i --out "out_dynaBARN"
        sleep 4
    done
done