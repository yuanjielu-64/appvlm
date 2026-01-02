#!/bin/bash

killall -9 rosmaster

killall gzclient

killall gzserver

for i in {279..8} ; do
    for j in {1..3} ; do            
        python evaluate_qwen_single.py --world_idx $i
        sleep 4
    done
done
