#!/bin/bash
for Q in {1..3}
do
    python3 ../bow_model.py $Q $1 -i ../data
    FILE_NAME=model_question_$Q
    tar -czf $1/$FILE_NAME.tar.gz  $1/$FILE_NAME
done
