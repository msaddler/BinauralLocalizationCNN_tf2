#!/bin/bash
#
#SBATCH --job-name=phaselocknet_train
#SBATCH --out="zslurm-%A_%a.out"
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --gres=gpu:1 --constraint=any-A100
##SBATCH --constraint=30GB
##SBATCH --gres=gpu:A100:1
##SBATCH --gres=gpu:A100-MCDERMOTT:2
##SBATCH --gres=gpu:RTXA6000:1
##SBATCH --gres=gpu:QUADRORTX6000:2
##SBATCH --gres=gpu:tesla-v100:2
##SBATCH --gres=gpu:GEFORCEGTX1080TI:4
#SBATCH --time=7-00:00:00
#SBATCH --array=20-39
##SBATCH --partition=mcdermott
##SBATCH --partition=use-everything
##SBATCH --nodelist=node093
##SBATCH --nodelist=dgx002
##SBATCH --exclude=dgx002
#SBATCH --dependency=afterok:28711303_503
#SBATCH --requeue

offset=0
job_idx=$(($SLURM_ARRAY_TASK_ID + $offset))

declare -a list_model_dir=(
    "/saved_models/FLDv00v01/sigmoid_rate_level_IHC0050Hz/archFrancl01"
    "/saved_models/FLDv00v01/sigmoid_rate_level_IHC0050Hz/archFrancl02"
    "/saved_models/FLDv00v01/sigmoid_rate_level_IHC0050Hz/archFrancl03"
    "/saved_models/FLDv00v01/sigmoid_rate_level_IHC0050Hz/archFrancl04"
    "/saved_models/FLDv00v01/sigmoid_rate_level_IHC0050Hz/archFrancl05"
    "/saved_models/FLDv00v01/sigmoid_rate_level_IHC0050Hz/archFrancl06"
    "/saved_models/FLDv00v01/sigmoid_rate_level_IHC0050Hz/archFrancl07"
    "/saved_models/FLDv00v01/sigmoid_rate_level_IHC0050Hz/archFrancl08"
    "/saved_models/FLDv00v01/sigmoid_rate_level_IHC0050Hz/archFrancl09"
    "/saved_models/FLDv00v01/sigmoid_rate_level_IHC0050Hz/archFrancl10"

    "/saved_models/FLDv00v01/sigmoid_rate_level_IHC3000Hz/archFrancl01"
    "/saved_models/FLDv00v01/sigmoid_rate_level_IHC3000Hz/archFrancl02"
    "/saved_models/FLDv00v01/sigmoid_rate_level_IHC3000Hz/archFrancl03"
    "/saved_models/FLDv00v01/sigmoid_rate_level_IHC3000Hz/archFrancl04"
    "/saved_models/FLDv00v01/sigmoid_rate_level_IHC3000Hz/archFrancl05"
    "/saved_models/FLDv00v01/sigmoid_rate_level_IHC3000Hz/archFrancl06"
    "/saved_models/FLDv00v01/sigmoid_rate_level_IHC3000Hz/archFrancl07"
    "/saved_models/FLDv00v01/sigmoid_rate_level_IHC3000Hz/archFrancl08"
    "/saved_models/FLDv00v01/sigmoid_rate_level_IHC3000Hz/archFrancl09"
    "/saved_models/FLDv00v01/sigmoid_rate_level_IHC3000Hz/archFrancl10"
    
    "/saved_models/augmented_2022JAN/taskSW/sigmoid_rate_level_IHC0050Hz/arch0_0000"
    "/saved_models/augmented_2022JAN/taskSW/sigmoid_rate_level_IHC0050Hz/arch0_0001"
    "/saved_models/augmented_2022JAN/taskSW/sigmoid_rate_level_IHC0050Hz/arch0_0002"
    "/saved_models/augmented_2022JAN/taskSW/sigmoid_rate_level_IHC0050Hz/arch0_0004"
    "/saved_models/augmented_2022JAN/taskSW/sigmoid_rate_level_IHC0050Hz/arch0_0006"
    "/saved_models/augmented_2022JAN/taskSW/sigmoid_rate_level_IHC0050Hz/arch0_0007"
    "/saved_models/augmented_2022JAN/taskSW/sigmoid_rate_level_IHC0050Hz/arch0_0008"
    "/saved_models/augmented_2022JAN/taskSW/sigmoid_rate_level_IHC0050Hz/arch0_0009"
    "/saved_models/augmented_2022JAN/taskSW/sigmoid_rate_level_IHC0050Hz/arch0_0016"
    "/saved_models/augmented_2022JAN/taskSW/sigmoid_rate_level_IHC0050Hz/arch0_0017"
    
    "/saved_models/augmented_2022JAN/taskSW/sigmoid_rate_level_IHC3000Hz/arch0_0000"
    "/saved_models/augmented_2022JAN/taskSW/sigmoid_rate_level_IHC3000Hz/arch0_0001"
    "/saved_models/augmented_2022JAN/taskSW/sigmoid_rate_level_IHC3000Hz/arch0_0002"
    "/saved_models/augmented_2022JAN/taskSW/sigmoid_rate_level_IHC3000Hz/arch0_0004"
    "/saved_models/augmented_2022JAN/taskSW/sigmoid_rate_level_IHC3000Hz/arch0_0006"
    "/saved_models/augmented_2022JAN/taskSW/sigmoid_rate_level_IHC3000Hz/arch0_0007"
    "/saved_models/augmented_2022JAN/taskSW/sigmoid_rate_level_IHC3000Hz/arch0_0008"
    "/saved_models/augmented_2022JAN/taskSW/sigmoid_rate_level_IHC3000Hz/arch0_0009"
    "/saved_models/augmented_2022JAN/taskSW/sigmoid_rate_level_IHC3000Hz/arch0_0016"
    "/saved_models/augmented_2022JAN/taskSW/sigmoid_rate_level_IHC3000Hz/arch0_0017"

    "/saved_models/augmented_2022JAN/taskSW/cochlearn_roex_human_init/arch0_0000"
    "/saved_models/augmented_2022JAN/taskSW/cochlearn_roex_human_init/arch0_0001"
    "/saved_models/augmented_2022JAN/taskSW/cochlearn_roex_human_init/arch0_0002"
    "/saved_models/augmented_2022JAN/taskSW/cochlearn_roex_human_init/arch0_0004"
    "/saved_models/augmented_2022JAN/taskSW/cochlearn_roex_human_init/arch0_0006"
    "/saved_models/augmented_2022JAN/taskSW/cochlearn_roex_human_init/arch0_0007"
    "/saved_models/augmented_2022JAN/taskSW/cochlearn_roex_human_init/arch0_0008"
    "/saved_models/augmented_2022JAN/taskSW/cochlearn_roex_human_init/arch0_0009"
    "/saved_models/augmented_2022JAN/taskSW/cochlearn_roex_human_init/arch0_0016"
    "/saved_models/augmented_2022JAN/taskSW/cochlearn_roex_human_init/arch0_0017"

    "/saved_models/FLDv00v01/cochlearn_roex_human_init/archFrancl01"
    "/saved_models/FLDv00v01/cochlearn_roex_human_init/archFrancl02"
    "/saved_models/FLDv00v01/cochlearn_roex_human_init/archFrancl03"
    "/saved_models/FLDv00v01/cochlearn_roex_human_init/archFrancl04"
    "/saved_models/FLDv00v01/cochlearn_roex_human_init/archFrancl05"
    "/saved_models/FLDv00v01/cochlearn_roex_human_init/archFrancl06"
    "/saved_models/FLDv00v01/cochlearn_roex_human_init/archFrancl07"
    "/saved_models/FLDv00v01/cochlearn_roex_human_init/archFrancl08"
    "/saved_models/FLDv00v01/cochlearn_roex_human_init/archFrancl09"
    "/saved_models/FLDv00v01/cochlearn_roex_human_init/archFrancl10"

    "/saved_models/augmented_2022JAN/taskSW/cochlearn_IHC0050Hz/arch0_0000"
    "/saved_models/augmented_2022JAN/taskSW/cochlearn_IHC0050Hz/arch0_0001"
    "/saved_models/augmented_2022JAN/taskSW/cochlearn_IHC0050Hz/arch0_0002"
    "/saved_models/augmented_2022JAN/taskSW/cochlearn_IHC0050Hz/arch0_0004"
    "/saved_models/augmented_2022JAN/taskSW/cochlearn_IHC0050Hz/arch0_0006"
    "/saved_models/augmented_2022JAN/taskSW/cochlearn_IHC0050Hz/arch0_0007"
    "/saved_models/augmented_2022JAN/taskSW/cochlearn_IHC0050Hz/arch0_0008"
    "/saved_models/augmented_2022JAN/taskSW/cochlearn_IHC0050Hz/arch0_0009"
    "/saved_models/augmented_2022JAN/taskSW/cochlearn_IHC0050Hz/arch0_0016"
    "/saved_models/augmented_2022JAN/taskSW/cochlearn_IHC0050Hz/arch0_0017"

    "/saved_models/FLDv00v01/cochlearn_IHC0050Hz/archFrancl01"
    "/saved_models/FLDv00v01/cochlearn_IHC0050Hz/archFrancl02"
    "/saved_models/FLDv00v01/cochlearn_IHC0050Hz/archFrancl03"
    "/saved_models/FLDv00v01/cochlearn_IHC0050Hz/archFrancl04"
    "/saved_models/FLDv00v01/cochlearn_IHC0050Hz/archFrancl05"
    "/saved_models/FLDv00v01/cochlearn_IHC0050Hz/archFrancl06"
    "/saved_models/FLDv00v01/cochlearn_IHC0050Hz/archFrancl07"
    "/saved_models/FLDv00v01/cochlearn_IHC0050Hz/archFrancl08"
    "/saved_models/FLDv00v01/cochlearn_IHC0050Hz/archFrancl09"
    "/saved_models/FLDv00v01/cochlearn_IHC0050Hz/archFrancl10"

    "/saved_models/FLDv00v01/sigmoid_rate_level/archFrancl01"
    "/saved_models/FLDv00v01/sigmoid_rate_level/archFrancl02"
    "/saved_models/FLDv00v01/sigmoid_rate_level/archFrancl03"
    "/saved_models/FLDv00v01/sigmoid_rate_level/archFrancl04"
    "/saved_models/FLDv00v01/sigmoid_rate_level/archFrancl05"
    "/saved_models/FLDv00v01/sigmoid_rate_level/archFrancl06"
    "/saved_models/FLDv00v01/sigmoid_rate_level/archFrancl07"
    "/saved_models/FLDv00v01/sigmoid_rate_level/archFrancl08"
    "/saved_models/FLDv00v01/sigmoid_rate_level/archFrancl09"
    "/saved_models/FLDv00v01/sigmoid_rate_level/archFrancl10"

    "/saved_models/augmented_2022JAN/task_S/IHC0320Hz_anf384H160M096L/arch0_0000"
    "/saved_models/augmented_2022JAN/task_S/IHC0320Hz_anf384H160M096L/arch0_0001"
    "/saved_models/augmented_2022JAN/task_S/IHC0320Hz_anf384H160M096L/arch0_0002"
    "/saved_models/augmented_2022JAN/task_S/IHC0320Hz_anf384H160M096L/arch0_0004"
    "/saved_models/augmented_2022JAN/task_S/IHC0320Hz_anf384H160M096L/arch0_0006"
    "/saved_models/augmented_2022JAN/task_S/IHC0320Hz_anf384H160M096L/arch0_0007"
    "/saved_models/augmented_2022JAN/task_S/IHC0320Hz_anf384H160M096L/arch0_0008"
    "/saved_models/augmented_2022JAN/task_S/IHC0320Hz_anf384H160M096L/arch0_0009"
    "/saved_models/augmented_2022JAN/task_S/IHC0320Hz_anf384H160M096L/arch0_0016"
    "/saved_models/augmented_2022JAN/task_S/IHC0320Hz_anf384H160M096L/arch0_0017"
    
    "/saved_models/augmented_2022JAN/task_S/IHC1000Hz_anf384H160M096L/arch0_0000"
    "/saved_models/augmented_2022JAN/task_S/IHC1000Hz_anf384H160M096L/arch0_0001"
    "/saved_models/augmented_2022JAN/task_S/IHC1000Hz_anf384H160M096L/arch0_0002"
    "/saved_models/augmented_2022JAN/task_S/IHC1000Hz_anf384H160M096L/arch0_0004"
    "/saved_models/augmented_2022JAN/task_S/IHC1000Hz_anf384H160M096L/arch0_0006"
    "/saved_models/augmented_2022JAN/task_S/IHC1000Hz_anf384H160M096L/arch0_0007"
    "/saved_models/augmented_2022JAN/task_S/IHC1000Hz_anf384H160M096L/arch0_0008"
    "/saved_models/augmented_2022JAN/task_S/IHC1000Hz_anf384H160M096L/arch0_0009"
    "/saved_models/augmented_2022JAN/task_S/IHC1000Hz_anf384H160M096L/arch0_0016"
    "/saved_models/augmented_2022JAN/task_S/IHC1000Hz_anf384H160M096L/arch0_0017"
    
    "/saved_models/augmented_2022JAN/task_W/IHC0320Hz_anf384H160M096L/arch0_0000"
    "/saved_models/augmented_2022JAN/task_W/IHC0320Hz_anf384H160M096L/arch0_0001"
    "/saved_models/augmented_2022JAN/task_W/IHC0320Hz_anf384H160M096L/arch0_0002"
    "/saved_models/augmented_2022JAN/task_W/IHC0320Hz_anf384H160M096L/arch0_0004"
    "/saved_models/augmented_2022JAN/task_W/IHC0320Hz_anf384H160M096L/arch0_0006"
    "/saved_models/augmented_2022JAN/task_W/IHC0320Hz_anf384H160M096L/arch0_0007"
    "/saved_models/augmented_2022JAN/task_W/IHC0320Hz_anf384H160M096L/arch0_0008"
    "/saved_models/augmented_2022JAN/task_W/IHC0320Hz_anf384H160M096L/arch0_0009"
    "/saved_models/augmented_2022JAN/task_W/IHC0320Hz_anf384H160M096L/arch0_0016"
    "/saved_models/augmented_2022JAN/task_W/IHC0320Hz_anf384H160M096L/arch0_0017"
    
    "/saved_models/augmented_2022JAN/task_W/IHC1000Hz_anf384H160M096L/arch0_0000"
    "/saved_models/augmented_2022JAN/task_W/IHC1000Hz_anf384H160M096L/arch0_0001"
    "/saved_models/augmented_2022JAN/task_W/IHC1000Hz_anf384H160M096L/arch0_0002"
    "/saved_models/augmented_2022JAN/task_W/IHC1000Hz_anf384H160M096L/arch0_0004"
    "/saved_models/augmented_2022JAN/task_W/IHC1000Hz_anf384H160M096L/arch0_0006"
    "/saved_models/augmented_2022JAN/task_W/IHC1000Hz_anf384H160M096L/arch0_0007"
    "/saved_models/augmented_2022JAN/task_W/IHC1000Hz_anf384H160M096L/arch0_0008"
    "/saved_models/augmented_2022JAN/task_W/IHC1000Hz_anf384H160M096L/arch0_0009"
    "/saved_models/augmented_2022JAN/task_W/IHC1000Hz_anf384H160M096L/arch0_0016"
    "/saved_models/augmented_2022JAN/task_W/IHC1000Hz_anf384H160M096L/arch0_0017"
    
    "/saved_models/FLDv00v01/cochlearn_float32/archFrancl01"
    "/saved_models/FLDv00v01/cochlearn_float32/archFrancl02"
    "/saved_models/FLDv00v01/cochlearn_float32/archFrancl03"
    "/saved_models/FLDv00v01/cochlearn_float32/archFrancl04"
    "/saved_models/FLDv00v01/cochlearn_float32/archFrancl05"
    "/saved_models/FLDv00v01/cochlearn_float32/archFrancl06"
    "/saved_models/FLDv00v01/cochlearn_float32/archFrancl07"
    "/saved_models/FLDv00v01/cochlearn_float32/archFrancl08"
    "/saved_models/FLDv00v01/cochlearn_float32/archFrancl09"
    "/saved_models/FLDv00v01/cochlearn_float32/archFrancl10"
    
    "/saved_models/augmented_2022JAN/taskSW/sigmoid_rate_level/arch0_0000"
    "/saved_models/augmented_2022JAN/taskSW/sigmoid_rate_level/arch0_0001"
    "/saved_models/augmented_2022JAN/taskSW/sigmoid_rate_level/arch0_0002"
    "/saved_models/augmented_2022JAN/taskSW/sigmoid_rate_level/arch0_0004"
    "/saved_models/augmented_2022JAN/taskSW/sigmoid_rate_level/arch0_0006"
    "/saved_models/augmented_2022JAN/taskSW/sigmoid_rate_level/arch0_0007"
    "/saved_models/augmented_2022JAN/taskSW/sigmoid_rate_level/arch0_0008"
    "/saved_models/augmented_2022JAN/taskSW/sigmoid_rate_level/arch0_0009"
    "/saved_models/augmented_2022JAN/taskSW/sigmoid_rate_level/arch0_0016"
    "/saved_models/augmented_2022JAN/taskSW/sigmoid_rate_level/arch0_0017"
    
    "/saved_models/augmented_2022JAN/taskSW/cochlearn/arch0_0000"
    "/saved_models/augmented_2022JAN/taskSW/cochlearn/arch0_0001"
    "/saved_models/augmented_2022JAN/taskSW/cochlearn/arch0_0002"
    "/saved_models/augmented_2022JAN/taskSW/cochlearn/arch0_0004"
    "/saved_models/augmented_2022JAN/taskSW/cochlearn/arch0_0006"
    "/saved_models/augmented_2022JAN/taskSW/cochlearn/arch0_0007"
    "/saved_models/augmented_2022JAN/taskSW/cochlearn/arch0_0008"
    "/saved_models/augmented_2022JAN/taskSW/cochlearn/arch0_0009"
    "/saved_models/augmented_2022JAN/taskSW/cochlearn/arch0_0016"
    "/saved_models/augmented_2022JAN/taskSW/cochlearn/arch0_0017"
    "/saved_models/augmented_2022JAN/taskASW/cochlearn/arch0_0000"
    "/saved_models/augmented_2022JAN/taskASW/cochlearn/arch0_0001"
    "/saved_models/augmented_2022JAN/taskASW/cochlearn/arch0_0002"
    "/saved_models/augmented_2022JAN/taskASW/cochlearn/arch0_0004"
    "/saved_models/augmented_2022JAN/taskASW/cochlearn/arch0_0006"
    "/saved_models/augmented_2022JAN/taskASW/cochlearn/arch0_0007"
    "/saved_models/augmented_2022JAN/taskASW/cochlearn/arch0_0008"
    "/saved_models/augmented_2022JAN/taskASW/cochlearn/arch0_0009"
    "/saved_models/augmented_2022JAN/taskASW/cochlearn/arch0_0016"
    "/saved_models/augmented_2022JAN/taskASW/cochlearn/arch0_0017"
    "/saved_models/augmented_2022JAN/task_S/cochlearn/arch0_0000"
    "/saved_models/augmented_2022JAN/task_S/cochlearn/arch0_0001"
    "/saved_models/augmented_2022JAN/task_S/cochlearn/arch0_0002"
    "/saved_models/augmented_2022JAN/task_S/cochlearn/arch0_0004"
    "/saved_models/augmented_2022JAN/task_S/cochlearn/arch0_0006"
    "/saved_models/augmented_2022JAN/task_S/cochlearn/arch0_0007"
    "/saved_models/augmented_2022JAN/task_S/cochlearn/arch0_0008"
    "/saved_models/augmented_2022JAN/task_S/cochlearn/arch0_0009"
    "/saved_models/augmented_2022JAN/task_S/cochlearn/arch0_0016"
    "/saved_models/augmented_2022JAN/task_S/cochlearn/arch0_0017"
    "/saved_models/augmented_2022JAN/task_W/cochlearn/arch0_0000"
    "/saved_models/augmented_2022JAN/task_W/cochlearn/arch0_0001"
    "/saved_models/augmented_2022JAN/task_W/cochlearn/arch0_0002"
    "/saved_models/augmented_2022JAN/task_W/cochlearn/arch0_0004"
    "/saved_models/augmented_2022JAN/task_W/cochlearn/arch0_0006"
    "/saved_models/augmented_2022JAN/task_W/cochlearn/arch0_0007"
    "/saved_models/augmented_2022JAN/task_W/cochlearn/arch0_0008"
    "/saved_models/augmented_2022JAN/task_W/cochlearn/arch0_0009"
    "/saved_models/augmented_2022JAN/task_W/cochlearn/arch0_0016"
    "/saved_models/augmented_2022JAN/task_W/cochlearn/arch0_0017"
)
model_dir=${list_model_dir[$job_idx]}
echo $HOSTNAME $job_idx $model_dir

## Choose DATA_TAG based on model_dir
SCRATCH_PATH_TO_USE=$SCRATCH_PATH
DATA_TAG="sr10000_cf050_nst0?0_BW10eN1_cohc10eN1_cihc10eN1_IHC3000Hz_dbspl030to090"
if [[ "$model_dir" == *"IHC3000Hz"* ]]; then
  DATA_TAG="sr10000_cf050_nst0?0_BW10eN1_cohc10eN1_cihc10eN1_IHC3000Hz_dbspl030to090"
  SCRATCH_PATH_TO_USE=$VAST_SCRATCH_PATH
fi
if [[ "$model_dir" == *"IHC1000Hz"* ]]; then
  DATA_TAG="sr10000_cf050_nst0?0_BW10eN1_cohc10eN1_cihc10eN1_IHC1000Hz_dbspl030to090"
fi
if [[ "$model_dir" == *"IHC0320Hz"* ]]; then
  DATA_TAG="sr10000_cf050_nst0?0_BW10eN1_cohc10eN1_cihc10eN1_IHC0320Hz_dbspl030to090"
fi
if [[ "$model_dir" == *"IHC0050Hz"* ]]; then
  DATA_TAG="sr10000_cf050_nst0?0_BW10eN1_cohc10eN1_cihc10eN1_IHC0050Hz_dbspl030to090"
  SCRATCH_PATH_TO_USE=$VAST_SCRATCH_PATH
fi
if [[ "$model_dir" == *"BW30eN1"* ]]; then
  DATA_TAG="sr10000_cf050_nst0?0_BW30eN1_cohc10eN1_cihc10eN1_IHC3000Hz_dbspl030to090"
fi

## Choose dataset based on model_dir
regex_train="$SCRATCH_PATH_TO_USE/data_WSN/JSIN_all_v3_augmented_2022SEP/train/$DATA_TAG/*tfrecords"
regex_valid="$SCRATCH_PATH_TO_USE/data_WSN/JSIN_all_v3_augmented_2022SEP/valid/$DATA_TAG/*tfrecords"
if [[ "$model_dir" == *"augmented_2022DEC"* ]]; then
    regex_train="$SCRATCH_PATH/data_WSN/JSIN_all_v3_augmented_2022DEC/train/$DATA_TAG/*tfrecords"
    regex_valid="$SCRATCH_PATH/data_WSN/JSIN_all_v3_augmented_2022DEC/valid/$DATA_TAG/*tfrecords"
fi
if [[ "$model_dir" == *"augmented_2022JAN"* ]]; then
    regex_train="$SCRATCH_PATH/data_WSN/JSIN_all_v3_augmented_2022JAN/train/$DATA_TAG/*tfrecords"
    regex_valid="$SCRATCH_PATH/data_WSN/JSIN_all_v3_augmented_2022JAN/valid/$DATA_TAG/*tfrecords"
fi

if [[ "$model_dir" == *"FLDv0"* ]]; then
    # Choose sound localization dataset
    DATA_TAG="sr10000_cf050_nst000_BW10eN1_cohc10eN1_cihc10eN1_IHC3000Hz_dbspl030to090"
    if [[ "$model_dir" == *"IHC3000Hz"* ]]; then
        DATA_TAG="sr10000_cf050_nst000_BW10eN1_cohc10eN1_cihc10eN1_IHC3000Hz_dbspl030to090"
    fi
    if [[ "$model_dir" == *"IHC1000Hz"* ]]; then
        DATA_TAG="sr10000_cf050_nst000_BW10eN1_cohc10eN1_cihc10eN1_IHC1000Hz_dbspl030to090"
    fi
    if [[ "$model_dir" == *"IHC0320Hz"* ]]; then
        DATA_TAG="sr10000_cf050_nst000_BW10eN1_cohc10eN1_cihc10eN1_IHC0320Hz_dbspl030to090"
    fi
    if [[ "$model_dir" == *"IHC0050Hz"* ]]; then
        DATA_TAG="sr10000_cf050_nst000_BW10eN1_cohc10eN1_cihc10eN1_IHC0050Hz_dbspl030to090"
    fi
    regex_train="$SCRATCH_PATH/data_localize/FLDv00v01/train/$DATA_TAG/*tfrecords"
    regex_valid="$SCRATCH_PATH/data_localize/FLDv00v01/valid/$DATA_TAG/*tfrecords"
fi

mixed_precision=1
if [[ "$model_dir" == *"float32"* ]]; then
    mixed_precision=0
    echo "Found 'float32' in model directory name --> mixed_precision=$mixed_precision"
fi

# export TF_FORCE_GPU_ALLOW_GROWTH=true
# export TF_CUDNN_WORKSPACE_LIMIT_IN_MB=2048
# export TF_FORCE_GPU_ALLOW_GROWTH=
# export TF_CUDNN_WORKSPACE_LIMIT_IN_MB=
singularity exec --nv \
-B /home \
-B $SCRATCH_PATH \
-B /nese \
-B /scratch2 \
-B /om2 \
-B /om4 \
-B /om2/user/msaddler/tfauditoryutil/saved_models:/saved_models \
-B /om2/user/msaddler/python-packages:/python-packages \
/om2/user/msaddler/vagrant/tensorflow-2.10.0.simg \
python -u phaselocknet_run.py \
-m "$model_dir" \
-c "config.json" \
-a "arch.json" \
-t "$regex_train" \
-v "$regex_valid" \
-mp $mixed_precision \
2>&1 | tee "/om2/user/msaddler/tfauditoryutil/$model_dir/log_optimize.out"
