#!/bin/bash

killall -9 rosmaster
killall gzclient
killall gzserver

# 你想要运行的 i 值
i_values=(299 294 288 284 283 282 281 278 276 275 264 258 245 237 228 197 182 167 138 129 111 99 58 33 19 )

for i in "${i_values[@]}"; do
    for j in {1..2}; do            
        python run_ddp.py --world_idx $i --out "test.txt"
        sleep 2
    done
done
