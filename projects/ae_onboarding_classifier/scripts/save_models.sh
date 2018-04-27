#!/bin/bash
# Used for saving models in a SageMaker compatible directory structure
for Q in {1..3}
do
    python3 ../bow_model.py $Q $1 -i ../data
    FILE_NAME=model_question_$Q
    EXPORT=$1/export
    mkdir $EXPORT
    mv $1/Servo $EXPORT
    tar -czf $1/$FILE_NAME.tar.gz $EXPORT
    rm -rf $EXPORT
done
