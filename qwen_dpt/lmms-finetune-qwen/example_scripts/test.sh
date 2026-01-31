NUM_GPUS=1


# ============================================================
# TEST CONFIGURATION FOR 3080 10GB GPU
# Only trains DPT head, freezes all other parameters
# ============================================================
PLANNER=ddp                                                      # planner name: dwa | teb | mppi | ddp
DATA_ROOT=/home/yuanjielu/robot_navigation/noetic/app_data       # TODO: CHANGE THIS to your local data path
MODEL_ID=qwen2.5-vl-regression                                   # model id for regression task (must match registered loader/collator)
CUSTOM_NAME="test_3080"                                          # test run on 3080

# Auto-generated paths based on PLANNER
TRAIN_DATA_PATH=${DATA_ROOT}/${PLANNER}_heurstic/splits_200k/chunk_000.json    # path to the training data json file
IMAGE_FOLDER=${DATA_ROOT}/${PLANNER}_heurstic                    # path to the image root folder; if provided, the image paths in the json should be relative
VIDEO_FOLDER=./example_data/videos                               # path to the video root folder; if provided, the video paths in the json should be relative
NUM_FRAMES=8                                                     # how many frames are sampled from each video

TRAIN_VISION_ENCODER=False                              # FREEZE vision encoder (save memory)
USE_VISION_LORA=False                                   # no lora for vision
TRAIN_VISION_PROJECTOR=False                            # FREEZE vision projector (save memory)

USE_LORA=False                                          # FREEZE LLM (save memory) - only train DPT head
Q_LORA=False                                            # not needed when USE_LORA=False
LORA_R=32                                                # not used
LORA_ALPHA=16                                            # not used

# Generate RUN_ID with optional custom name
if [ -z "$CUSTOM_NAME" ]; then
    RUN_ID=${MODEL_ID}_lora-${USE_LORA}_${PLANNER}_regression
else
    RUN_ID=${CUSTOM_NAME}
fi

OUTPUT_DIR=../../ros_jackal/model/${PLANNER}/${RUN_ID}          # output directory under ros_jackal/model/{planner}/

DS_STAGE=zero2                                          # zero2 for faster initialization on single GPU
PER_DEVICE_BATCH_SIZE=1                                 # small batch size for 10GB GPU
GRAD_ACCUM=8                                            # maintain effective batch size = 8
NUM_EPOCHS=1                                            # just 1 epoch for testing

LR=1e-4                                                 # higher LR for DPT head only training
MODEL_MAX_LEN=1024                                      # reduce sequence length to save memory


torchrun \
    --nnodes=1 \
    --nproc_per_node ${NUM_GPUS} \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:0 \
     train_regression.py \
    --model_id $MODEL_ID \
    --planner $PLANNER \
    --data_path $TRAIN_DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --video_folder $VIDEO_FOLDER \
    --num_frames $NUM_FRAMES \
    --output_dir $OUTPUT_DIR \
    --report_to wandb \
    --run_name $RUN_ID \
    --deepspeed ./ds_configs/${DS_STAGE}.json \
    --bf16 True \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --save_strategy "steps" \
    --save_steps 2500 \
    --learning_rate ${LR} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length $MODEL_MAX_LEN \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --train_vision_encoder $TRAIN_VISION_ENCODER \
    --use_vision_lora $USE_VISION_LORA \
    --train_vision_projector $TRAIN_VISION_PROJECTOR \
    --use_lora $USE_LORA \
    --q_lora $Q_LORA \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA

