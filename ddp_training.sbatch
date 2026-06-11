#!/bin/bash
#SBATCH --job-name=ddp_train

##############################################################
# CLUSTER RESOURCES
##############################################################

#SBATCH --partition=gpu_a100
#SBATCH --nodes=2

#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4

# CPU cores per task
#SBATCH --cpus-per-task=18
#SBATCH --time=72:00:00

##############################################################
# TRAINING CONFIG
##############################################################

CONFIG=$1
export NUM_WORKERS=16

echo "============================================================"
echo "JOB START"
echo "============================================================"

date

##############################################################
# SLURM DEBUG INFO
##############################################################

echo ""
echo "============================================================"
echo "SLURM VARIABLES"
echo "============================================================"

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "SLURM_NNODES=$SLURM_NNODES"
echo "SLURM_NTASKS=$SLURM_NTASKS"

echo ""
echo "Allocated nodes:"
scontrol show hostnames "$SLURM_JOB_NODELIST"

##############################################################
# LOAD MODULES
##############################################################

echo ""
echo "============================================================"
echo "LOADING MODULES"
echo "============================================================"

module purge

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

echo ""
echo "Loaded modules:"
module list

##############################################################
# ACTIVATE ENVIRONMENT
##############################################################

echo ""
echo "============================================================"
echo "ACTIVATING PYTHON ENVIRONMENT"
echo "============================================================"

source "$HOME/venvs/mlr/bin/activate"

echo "Python executable:"
which python

echo ""
echo "Python version:"
python --version

##############################################################
# GPU DEBUGGING
##############################################################

echo ""
echo "============================================================"
echo "GPU INFO"
echo "============================================================"

nvidia-smi

##############################################################
# DISTRIBUTED VARIABLES
##############################################################

echo ""
echo "============================================================"
echo "SETTING DISTRIBUTED VARIABLES"
echo "============================================================"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

MASTER_PORT=29500

WORLD_SIZE=$((SLURM_NNODES * 4))

export MASTER_ADDR
export MASTER_PORT
export WORLD_SIZE

echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "WORLD_SIZE=$WORLD_SIZE"

echo ""
echo "============================================================"
echo "TRAINING CONFIGURATION"
echo "============================================================"

echo "CONFIG=$CONFIG"

##############################################################
# RUN TRAINING
##############################################################

echo ""
echo "============================================================"
echo "STARTING TRAINING"
echo "============================================================"

srun python3 train.py --config "$CONFIG" || true

echo ""
echo "============================================================"
echo "JOB FINISHED"
echo "============================================================"

date
