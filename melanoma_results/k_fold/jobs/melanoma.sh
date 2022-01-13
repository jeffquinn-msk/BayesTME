#!/usr/bin/env bash
#BSUB -W 96:00
#BSUB -R rusage[mem=24]
#BSUB -J melanoma[1-420]
#BSUB -e ../setup/outputs/melanoma_%I.err
#BSUB -eo ../setup/outputs/melanoma_%I.out

cd $LS_SUBCWD
python grid_search_cfg.py --config ../setup/config/melanoma/config_${LSB_JOBINDEX}.cfg
