#!/bin/bash
#
if [ -z $1 ]; then
	echo "No result directory name given"
	exit 1
fi

STARTDIR=$(pwd)
SCRIPTDIR=$(dirname $0)
# Set to base evaluation directory (in Docker setup)
cd $SCRIPTDIR/../..

echo "Changing into: $(pwd)"

START=$(date +%s)
	python -m ot_backprop_pnwo.evaluation.evaluation_two_phase ot-wo-evaluation/data "ot-wo-evaluation/$1" PEMSC --repetitions 30 --poolSize 66 
	#python -m ot_backprop_pnwo.evaluation.evaluation_two_phase ot-wo-evaluation/data "ot-wo-evaluation/$1" EMSC --repetitions 30 --poolSize 66 
	#python -m ot_backprop_pnwo.evaluation.evaluation_two_phase ot-wo-evaluation/data "ot-wo-evaluation/$1" --repetitions 30 --poolSize 10
	END=$(date +%s)
	DIFF=$((END - START))
	echo "Ran evaluation in ${DIFF}s" 
	cp logfile.log "ot-wo-evaluation/$1"
cd $STARTDIR


