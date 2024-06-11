#!/bin/bash

PATH_LOG="ot-wo-evaluation/data/logs/road_fines/road_fines.xes"
PATH_MODEL="ot-wo-evaluation/data/models/road_fines/hot-start/rtfm_HM_FDE.pnml"
PATH_OUT="ot-wo-evaluation/test/rtfm_HM_FDE-single-run.pnml"


########################################
# Call
########################################
STARTDIR=$(pwd)
SCRIPTDIR=$(dirname $0)
# Set to base evaluation directory (in Docker setup)
cd $SCRIPTDIR/../..

echo "Changing into: $(pwd)"

START=$(date +%s)
python -m ot_backprop_pnwo.run_wawe "${PATH_LOG}" "${PATH_MODEL}" "${PATH_OUT}" 600 600 --warmStart
END=$(date +%s)

DIFF=$((END - START))
echo "Ran WAWE in ${DIFF}s" 

cd $STARTDIR

