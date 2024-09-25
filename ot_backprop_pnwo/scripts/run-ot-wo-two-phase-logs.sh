#!/bin/bash
#
if [ -z $1 ]; then
	echo "No result directory name given"
	exit 1
fi

########################################
# Arguments
########################################
LOG_PATH_BASE='logs'
SPN_PATH_BASE='models'
LOG_SPN_PAIRS=(
	"((\"RTFM\", \"${LOG_PATH_BASE}/road_fines/road_fines.xes\",  \"${SPN_PATH_BASE}/road_fines/cold-start\", False), "
	"(\"RTFM\", \"${LOG_PATH_BASE}/road_fines/road_fines.xes\",  \"${SPN_PATH_BASE}/road_fines/hot-start\", True), "
	"(\"BPIC18Ref\", \"${LOG_PATH_BASE}/bpic-2018-reference/bpic-2018-reference.xes\",  \"${SPN_PATH_BASE}/bpic-2018-reference/cold-start\", False),"
	"(\"BPIC18Ref\", \"${LOG_PATH_BASE}/bpic-2018-reference/bpic-2018-reference.xes\",  \"${SPN_PATH_BASE}/bpic-2018-reference/hot-start\", True))"
)

########################################
# Call
########################################
STARTDIR=$(pwd)
SCRIPTDIR=$(dirname $0)
# Set to base evaluation directory (in Docker setup)
cd $SCRIPTDIR/../..

echo "Changing into: $(pwd)"

START=$(date +%s)

# Replicate Phase 1
#python -m ot_backprop_pnwo.evaluation.evaluation_two_phase ot-wo-evaluation/data "ot-wo-evaluation/$1" PEMSC --residualHandling ADD_RESIDUAL_ELEMENT --repetitions 30 --poolSize 80 
# Test Residual Normalization
python -m ot_backprop_pnwo.evaluation.evaluation_two_phase ot-wo-evaluation/data "ot-wo-evaluation/$1" PEMSC --residualHandling NORMALIZE --repetitions 30 --poolSize 80 
#python -m ot_backprop_pnwo.evaluation.evaluation_two_phase ot-wo-evaluation/data/test-data "ot-wo-evaluation/$1" PEMSC --repetitions 2 --poolSize 6 --logNetData "${LOG_SPN_PAIRS[*]}" --otSizes 100,100 --enablePhaseII
#python -m ot_backprop_pnwo.evaluation.evaluation_two_phase ot-wo-evaluation/data "ot-wo-evaluation/$1" EMSC --repetitions 30 --poolSize 66 
#python -m ot_backprop_pnwo.evaluation.evaluation_two_phase ot-wo-evaluation/data "ot-wo-evaluation/$1" --repetitions 30 --poolSize 10

END=$(date +%s)
DIFF=$((END - START))
echo "Ran evaluation in ${DIFF}s" 
cp logfile.log "ot-wo-evaluation/$1"
cd $STARTDIR

