
export WORK=/projectnb/mlresearch/amartya/voice-sched/cache
export HF_HOME=$WORK/hf
export VLLM_CACHE_ROOT=$WORK/vllm
export FLASHINFER_WORKSPACE_BASE=$WORK
export TRITON_CACHE_DIR=$WORK/triton
export PIP_CACHE_DIR=$WORK/pip
export MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
module load python3/3.12.4 2>/dev/null || module load python3
source $WORK/venv/bin/activate 2>/dev/null || trueexport VLLM_USE_FLASHINFER_SAMPLER=0
module load cuda/12.8
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
