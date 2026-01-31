# SLURM 命令参考 (Rutgers ECS)

Rutgers University 集群作业调度系统参考文档
来源: https://ecs.rutgers.edu/slurm_commands.html

---

## 📝 作业提交

### sbatch - 提交批处理脚本
提交批处理脚本到集群队列

```bash
sbatch myscript.sh
```

**示例 SLURM 脚本** (`train_ddp.slurm`):
```bash
#!/bin/bash
#SBATCH --job-name=rlft_ddp           # 作业名称
#SBATCH --output=logs/train_%j.out    # 输出文件 (%j = JobID)
#SBATCH --error=logs/train_%j.err     # 错误文件
#SBATCH --time=24:00:00               # 时间限制 (HH:MM:SS)
#SBATCH --nodes=1                     # 节点数
#SBATCH --ntasks=1                    # 任务数
#SBATCH --cpus-per-task=16            # 每任务CPU数
#SBATCH --gres=gpu:3                  # GPU数量
#SBATCH --mem=64G                     # 内存
#SBATCH --partition=gpu               # 分区名称

# 加载环境
module load cuda/11.8
source activate lmms-finetune-qwen

# 运行训练
bash launch_ddp.sh 3 --config_file ddp --policy_name ddp_rlft
```

---

## ❌ 作业取消

### scancel - 取消作业
停止正在运行或排队的作业

```bash
# 取消单个作业
scancel <jobid>

# 取消用户所有作业
scancel -u <username>
scancel -u yl2832

# 按作业名称取消
scancel --name rlft_ddp

# 取消特定分区的作业
scancel -p gpu -u yl2832
```

### sdel - 删除作业和临时文件
删除作业并清理节点上创建的临时目录

```bash
sdel <jobid>
```

---

## 📊 作业监控

### squeue - 查看作业队列
查看调度队列中的作业状态

```bash
# 查看你的作业
squeue -u yl2832

# 查看特定作业
squeue -j 706

# 查看作业预计开始时间
squeue -j 706 --format="%S"

# 查看所有待运行作业的预计开始时间
squeue --start

# 自定义输出格式
squeue -u yl2832 --format="%.10i %.9P %.30j %.8T %.10M %.6D %R"
#   %i = JobID
#   %P = Partition
#   %j = JobName
#   %T = State (R=Running, PD=Pending)
#   %M = Time used
#   %D = Nodes
#   %R = Reason/NodeList
```

**输出示例**:
```
JOBID PARTITION     NAME     ST       TIME  NODES NODELIST(REASON)
12345 gpu         rlft_ddp   R    1:23:45      1 gpu001
12346 gpu         rlft_dwa  PD       0:00      1 (Resources)
```

### sqlog - 查看作业日志
查看运行中和已完成作业的信息

```bash
# 查看你的作业历史
sqlog -u yl2832

# 查看特定作业详情
sqlog -j 12345

# 查看最近N个作业
sqlog -u yl2832 -n 10
```

### sstat - 查看运行中作业的资源使用
实时监控正在运行作业的资源占用

```bash
# 查看作业资源使用
sstat -j <jobid>

# 查看内存使用
sstat --format="JobID,MaxRSS,AveCPU" -j <jobid>

# 详细资源统计
sstat -j <jobid> --format="JobID,MaxRSS,MaxVMSize,AveCPU,AveRSS"
```

**可用字段**:
- `MaxRSS` - 最大常驻内存
- `MaxVMSize` - 最大虚拟内存
- `AveCPU` - 平均CPU使用
- `AveRSS` - 平均常驻内存

### sacct - 查看作业统计（包括已完成）
查看已完成和运行中作业的资源统计

```bash
# 查看作业统计
sacct -j <jobid>

# 自定义输出格式
sacct --format="JobID,JobName,MaxRSS,Elapsed,State" -j <jobid>

# 查看最近N天的作业
sacct --starttime=2026-01-20 --format="JobID,JobName,State,Elapsed"

# 查看可用字段
sacct --helpformat
```

**常用字段**:
- `JobID` - 作业ID
- `JobName` - 作业名称
- `State` - 状态 (COMPLETED, FAILED, TIMEOUT, etc.)
- `Elapsed` - 运行时间
- `MaxRSS` - 最大内存使用
- `AllocCPUS` - 分配的CPU数
- `AllocGRES` - 分配的GPU等资源

**示例输出**:
```bash
sacct --format="JobID,JobName,State,Elapsed,MaxRSS,AllocGRES" -j 12345
```
```
JobID        JobName      State    Elapsed     MaxRSS  AllocGRES
12345      rlft_ddp  COMPLETED   06:23:45          0  gpu:3
12345.0       python  COMPLETED   06:23:45   45678901  gpu:3
```

---

## 🖥️ 系统信息

### sinfo - 查看节点和分区信息
查看集群节点状态和分区配置

```bash
# 查看所有分区
sinfo

# 查看特定分区
sinfo -p gpu

# 详细格式
sinfo --format="%P %a %l %D %t %N"
#   %P = Partition
#   %a = Availability
#   %l = TimeLimit
#   %D = Nodes
#   %t = State
#   %N = NodeList
```

**节点状态**:
- `idle` - 空闲
- `alloc` - 已分配
- `mix` - 部分使用
- `down` - 停机
- `drain` - 排空中

### scontrol - 显示详细配置
查看节点、分区、作业的详细信息

```bash
# 查看分区详情
scontrol show partition=gpu
scontrol show partition=SOE_main

# 查看节点详情
scontrol show node=gpu001
scontrol show node

# 查看作业详情
scontrol show job=12345

# 查看作业步骤
scontrol show step=12345.0
```

**示例输出** (show job):
```
JobId=12345 JobName=rlft_ddp
   UserId=yl2832(12345) GroupId=users(100)
   Priority=4294901743 Nice=0 Account=default QOS=normal
   JobState=RUNNING Reason=None Dependency=(null)
   Requeue=1 Restarts=0 BatchFlag=1 Reboot=0 ExitCode=0:0
   RunTime=01:23:45 TimeLimit=24:00:00 TimeMin=N/A
   SubmitTime=2026-01-29T10:00:00 EligibleTime=2026-01-29T10:00:00
   StartTime=2026-01-29T10:05:00 EndTime=2026-01-30T10:05:00
   Partition=gpu AllocNode:Sid=login01:12345
   ReqNodeList=(null) ExcNodeList=(null)
   NodeList=gpu001
   BatchHost=gpu001
   NumNodes=1 NumCPUs=16 NumTasks=1 CPUs/Task=16 ReqB:S:C:T=0:0:*:*
   TRES=cpu=16,mem=64G,node=1,gres/gpu=3
   Command=/path/to/train_ddp.slurm
   WorkDir=/path/to/workdir
   StdErr=/path/to/logs/train_12345.err
   StdOut=/path/to/logs/train_12345.out
```

### smap - 图形化查看
图形化查看作业、分区和配置

```bash
smap
```

---

## 🔧 实用技巧

### 1. 监控 GPU 使用
```bash
# 在运行节点上检查GPU
srun --jobid=12345 --pty nvidia-smi

# 持续监控
watch -n 1 'squeue -u yl2832 && nvidia-smi'
```

### 2. 交互式会话
```bash
# 申请交互式GPU节点
srun --partition=gpu --gres=gpu:1 --time=2:00:00 --pty bash

# 在交互式节点上运行命令
srun --jobid=12345 --pty bash
```

### 3. 作业数组（批量运行）
```bash
#!/bin/bash
#SBATCH --array=0-299%10    # 运行300个任务，每次最多10个并行

# 使用 $SLURM_ARRAY_TASK_ID 作为 world_idx
python evaluate.py --world_idx $SLURM_ARRAY_TASK_ID
```

### 4. 依赖关系
```bash
# 等待前一个作业完成后再运行
JOB1=$(sbatch train.slurm | awk '{print $4}')
sbatch --dependency=afterok:$JOB1 eval.slurm
```

### 5. 邮件通知
```bash
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your.email@rutgers.edu
```

---

## 📋 常用命令快速参考

| 命令 | 用途 | 示例 |
|------|------|------|
| `sbatch` | 提交作业 | `sbatch train.slurm` |
| `squeue -u USER` | 查看我的作业 | `squeue -u yl2832` |
| `scancel JOB` | 取消作业 | `scancel 12345` |
| `sinfo` | 查看节点 | `sinfo -p gpu` |
| `sacct -j JOB` | 查看作业统计 | `sacct -j 12345` |
| `scontrol show job JOB` | 作业详情 | `scontrol show job 12345` |

---

## 🎯 RLFT 训练的 SLURM 脚本模板

### 单节点多GPU训练 (DDP)
```bash
#!/bin/bash
#SBATCH --job-name=rlft_ddp_train
#SBATCH --output=logs/rlft_%j.out
#SBATCH --error=logs/rlft_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:3
#SBATCH --mem=128G
#SBATCH --partition=gpu

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# 加载模块
module load cuda/11.8
module load python/3.10

# 激活环境
source ~/miniconda3/bin/activate lmms-finetune-qwen

# 切换到工作目录
cd /data/local/yl2832/appvlm_ws/src/ros_jackal

# 运行训练
bash rlft/launch_ddp.sh 3 \
    --config_file ddp \
    --policy_name ddp_rlft \
    --skip_test

# 训练完成
echo "=========================================="
echo "Training completed at: $(date)"
echo "=========================================="
```

### 评估作业数组
```bash
#!/bin/bash
#SBATCH --job-name=eval_qwen
#SBATCH --output=logs/eval_%A_%a.out
#SBATCH --error=logs/eval_%A_%a.err
#SBATCH --array=0-299%20        # 300个world，每次20个并行
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=cpu

cd /data/local/yl2832/appvlm_ws/src/ros_jackal

# 使用数组任务ID作为world_idx
python script/qwen/evaluate_qwen_single.py \
    --world_idx $SLURM_ARRAY_TASK_ID \
    --qwen_url http://localhost:5000 \
    --policy_name ddp_qwen
```

---

## 📌 注意事项

1. **时间限制**: 合理设置 `--time`，过短会被kill，过长浪费资源
2. **内存估算**: DDP训练建议 32GB per GPU
3. **GPU类型**: 指定GPU型号 `--gres=gpu:a100:3` 或 `--gres=gpu:v100:2`
4. **分区选择**: 根据资源需求选择合适的partition
5. **输出目录**: 确保 logs 目录存在，否则作业会失败
6. **环境加载**: 在脚本中显式加载所有依赖的模块和环境

---

**最后更新**: 2026-01-29
**参考链接**: https://ecs.rutgers.edu/slurm_commands.html
