#!/bin/bash

#SBATCH --partition=normal
#SBATCH --job-name=GetResults
#SBATCH --exclude=amd057,amd058,amd059,amd060
#SBATCH --output=cpu_report/r-cpu-test-%j.out
#SBATCH --error=cpu_report/r-cpu-test-%j.err
#SBATCH --time=4-23:59:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1GB

SINGULARITY_BASE=/scratch/ylu22/appvlm/src/ros_jackal
SINGULARITY_IMG=$SINGULARITY_BASE/jackal.sif

export ROS_HOSTNAME=localhost
export ROS_MASTER_URI=http://localhost:11311
export ROS_LOG_DIR=/tmp

module load singularity

cd ../../..

ACTOR_ID=$1
MODEL_ID=$2
BASE_LINE=$3

./singularity_run.sh $SINGULARITY_IMG python3 script/generate_data.py --buffer_path buffer/ --id $ACTOR_ID --world_path jackal_helper/worlds/BARN/ --policy_name dwa_heurstic