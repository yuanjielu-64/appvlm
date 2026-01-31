NUM_GPUS=1
DISTRIBUTED_ARGS="
    --nnodes=1 \
    --nproc_per_node ${NUM_GPUS} \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:0
"

# arguments that are very likely to be changed
# according to your own case
PLANNER=mppi                                                  # planner name: dwa | teb | mppi | ddp
HEAD_TYPE=dpt                                                    # regression head type: simple_mlp | transformer | dpt
DATA_ROOT=/scratch/ylu22/app_data                                 # root directory for training data
#DATA_ROOT=/home/yuanjielu/robot_navigation/noetic/app_data
MODEL_ID=qwen2.5-vl-regression                                   # model id for regression task (must match registered loader/collator)
CUSTOM_NAME=""                                                   # optional custom name suffix for RUN_ID (leave empty for default)

# History frames configuration
NUM_HISTORY_FRAMES=2                                             # number of history frames to use (0 = disabled)
HISTORY_DIM=256                                                  # dimension of history encoder output (only used if NUM_HISTORY_FRAMES > 0)
HISTORY_IMAGE_SIZE=224                                           # image size for history frames (only used if NUM_HISTORY_FRAMES > 0)

# Auto-detect USE_HISTORY based on NUM_HISTORY_FRAMES
if [ "$NUM_HISTORY_FRAMES" -gt 0 ]; then
    USE_HISTORY=True
else
    USE_HISTORY=False
fi

# Label noise configuration (for regularization)
LABEL_NOISE_STD=0.01                                              # standard deviation of Gaussian noise to add to labels (0.0 = no noise)

# Checkpoint configuration (resume from previous training)
RESUME_FROM_CHECKPOINT=""                                         # path to checkpoint directory to resume from (leave empty to start from scratch)
# Example: RESUME_FROM_CHECKPOINT="../../ros_jackal/model/mppi/qwen2.5-vl-regression_lora-True_mppi_regression_1/checkpoint-2500"

# Auto-generated paths based on PLANNER
TRAIN_DATA_PATH=${DATA_ROOT}/${PLANNER}_heurstic/splits_200k/chunk_000.json    # path to the training data json file
EVAL_DATA_PATH=${DATA_ROOT}/${PLANNER}_heurstic/splits_200k/chunk_000.json     # path to the evaluation data json file
IMAGE_FOLDER=${DATA_ROOT}/${PLANNER}_heurstic                    # path to the image root folder; if provided, the image paths in the json should be relative
VIDEO_FOLDER=./example_data/videos                               # path to the video root folder; if provided, the video paths in the json should be relative
NUM_FRAMES=8                                                     # how many frames are sampled from each video

TRAIN_VISION_ENCODER=False                              # whether train the vision encoder
USE_VISION_LORA=False                                   # whether use lora for vision encoder (only effective when `TRAIN_VISION_ENCODER` is True)
TRAIN_VISION_PROJECTOR=False                            # whether train the vision projector (only full finetuning is supported)

USE_LORA=True                                           # whether use lora for llm
Q_LORA=False                                            # whether use q-lora for llm; only effective when `USE_LORA` is True
LORA_R=128                                                # the lora rank (both llm and vision encoder)
LORA_ALPHA=64                                            # the lora alpha (both llm and vision encoder)

# Generate RUN_ID with optional custom name
if [ -z "$CUSTOM_NAME" ]; then
    RUN_ID=${MODEL_ID}_lora-${USE_LORA}_${PLANNER}_regression_1
else
    RUN_ID=${CUSTOM_NAME}
fi

OUTPUT_DIR=../../ros_jackal/model/${PLANNER}/${RUN_ID}          # output directory under ros_jackal/model/{planner}/

DS_STAGE=zero2                                          # deepspeed stage; < zero2 | zero3 > (use zero2 to test LoRA save)
PER_DEVICE_BATCH_SIZE=8                                 # batch size per GPU
GRAD_ACCUM=1                                            # gradient accumulation steps
NUM_EPOCHS=1                                            # number of training epochs

LR=2e-5                                                 # learning rate
MODEL_MAX_LEN=1048                                      # maximum input length of the model
MAX_EVAL_SAMPLES=200                                    # max number of eval samples (set to empty for full eval)


# Build history-related arguments conditionally
if [ "$USE_HISTORY" = "True" ]; then
    HISTORY_ARGS="--use_history --num_history_frames $NUM_HISTORY_FRAMES --history_dim $HISTORY_DIM --history_image_size $HISTORY_IMAGE_SIZE"
else
    HISTORY_ARGS=""
fi

# Build checkpoint resume arguments conditionally
if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
    CHECKPOINT_ARGS="--resume_from_checkpoint $RESUME_FROM_CHECKPOINT"
else
    CHECKPOINT_ARGS=""
fi

torchrun $DISTRIBUTED_ARGS train_regression.py \
    --model_id $MODEL_ID \
    --planner $PLANNER \
    --head_type $HEAD_TYPE \
    $HISTORY_ARGS \
    $CHECKPOINT_ARGS \
    --label_noise_std $LABEL_NOISE_STD \
    --data_path $TRAIN_DATA_PATH \
    --eval_data_path $EVAL_DATA_PATH \
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
    --eval_strategy "steps" \
    --eval_steps 2500 \
    --max_eval_samples $MAX_EVAL_SAMPLES \
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

