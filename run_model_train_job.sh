#!/bin/bash
#
#SBATCH --job-name=localization_model_train
#SBATCH --out="slurm-%A_%a.out"
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-9
#SBATCH --partition=normal --time=2-0
#SBATCH --requeue

offset=0
job_idx=$(($SLURM_ARRAY_TASK_ID + $offset))

declare -a list_model_dir=(
    "models/msaddler_tf2_model_example/archFrancl01"
    "models/msaddler_tf2_model_example/archFrancl02"
    "models/msaddler_tf2_model_example/archFrancl03"
    "models/msaddler_tf2_model_example/archFrancl04"
    "models/msaddler_tf2_model_example/archFrancl05"
    "models/msaddler_tf2_model_example/archFrancl06"
    "models/msaddler_tf2_model_example/archFrancl07"
    "models/msaddler_tf2_model_example/archFrancl08"
    "models/msaddler_tf2_model_example/archFrancl09"
    "models/msaddler_tf2_model_example/archFrancl10"
)
model_dir=${list_model_dir[$job_idx]}
echo $HOSTNAME $job_idx $model_dir

regex_train="/om2/scratch/*/msaddler/data_localize/dataset_localization/v01/train/sr10000_cf050_nst000_BW10eN1_cohc10eN1_cihc10eN1_IHC3000Hz_dbspl030to090/*tfrecords"
regex_valid="/om2/scratch/*/msaddler/data_localize/dataset_localization/v01/valid/sr10000_cf050_nst000_BW10eN1_cohc10eN1_cihc10eN1_IHC3000Hz_dbspl030to090/*tfrecords"
mixed_precision=1

# Activate python environment and run `phaselocknet_run.py`
module add openmind8/anaconda
source activate tf
python -u run_model.py \
-m "$model_dir" \
-c "config.json" \
-a "arch.json" \
-t "$regex_train" \
-v "$regex_valid" \
-mp $mixed_precision \
2>&1 | tee "$model_dir/log_optimize.out"
