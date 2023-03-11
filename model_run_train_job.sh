#!/bin/bash
#
#SBATCH --job-name=localization_model_train
#SBATCH --out="slurm-%A_%a.out"
#SBATCH --cpus-per-task=20
#SBATCH --mem=48G
#SBATCH --gres=gpu:tesla-v100:2
#SBATCH --time=7-00:00:00
#SBATCH --array=0-3
#SBATCH --partition=mcdermott
##SBATCH --partition=use-everything
#SBATCH --requeue

offset=0
job_idx=$(($SLURM_ARRAY_TASK_ID + $offset))

declare -a list_model_dir=(
    "/saved_models/msaddler_tf2_model_example/archFrancl01"
    "/saved_models/msaddler_tf2_model_example/archFrancl02"
    "/saved_models/msaddler_tf2_model_example/archFrancl03"
    "/saved_models/msaddler_tf2_model_example/archFrancl04"
    "/saved_models/msaddler_tf2_model_example/archFrancl05"
    "/saved_models/msaddler_tf2_model_example/archFrancl06"
    "/saved_models/msaddler_tf2_model_example/archFrancl07"
    "/saved_models/msaddler_tf2_model_example/archFrancl08"
    "/saved_models/msaddler_tf2_model_example/archFrancl09"
    "/saved_models/msaddler_tf2_model_example/archFrancl10"
)
model_dir=${list_model_dir[$job_idx]}
echo $HOSTNAME $job_idx $model_dir

regex_train="/net/weka-nfs.ib.cluster/scratch/scratch/*/msaddler/data_localize/FLDv00v01/train/sr10000_cf050_nst000_BW10eN1_cohc10eN1_cihc10eN1_IHC3000Hz_dbspl030to090/*tfrecords"
regex_valid="/net/weka-nfs.ib.cluster/scratch/scratch/*/msaddler/data_localize/FLDv00v01/valid/sr10000_cf050_nst000_BW10eN1_cohc10eN1_cihc10eN1_IHC3000Hz_dbspl030to090/*tfrecords"
mixed_precision=1


singularity exec --nv \
-B /home \
-B /net/weka-nfs.ib.cluster/scratch \
-B /net/vast-storage/scratch \
-B /nese \
-B /scratch2 \
-B /om2 \
-B /om4 \
-B /om2/user/msaddler/francl_mcdermott_tf2/saved_models:/saved_models \
/om2/user/msaddler/vagrant/tensorflow-2.10.0.simg \
python -u model_run.py \
-m "$model_dir" \
-c "config.json" \
-a "arch.json" \
-t "$regex_train" \
-v "$regex_valid" \
-mp $mixed_precision \
2>&1 | tee "/om2/user/msaddler/francl_mcdermott_tf2/$model_dir/log_optimize.out"
