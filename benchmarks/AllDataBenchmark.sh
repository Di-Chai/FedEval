#!/usr/bin/env bash

cd ..

if [ $FEDEVAL_PYTHON ];then
	echo "Python = $FEDEVAL_PYTHON"
else
  export FEDEVAL_PYTHON="$(which python)"
  echo "Python not provided using $FEDEVAL_PYTHON"
fi

echo -n "Strategy Name: "
read STRATEGY

echo -n "Non-IID?: "
read NONIID

export FED_EVAL_REPEAT=10
export FED_EVAL_LOG_DIR=log/JMLRSummary

for DATESET in mnist femnist celeba semantic140 shakespeare
do
    if [ $NONIID = "True" ]
    then
      if [$DATESET = 'mnist']
      then
        $FEDEVAL_PYTHON trial.py -s $STRATEGY -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -n 1 -e run -d $DATESET
      else
        $FEDEVAL_PYTHON trial.py -s $STRATEGY -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -e run -d $DATESET
      fi
    else
      $FEDEVAL_PYTHON trial.py -s $STRATEGY -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i false -e run -d $DATESET
    fi
done
