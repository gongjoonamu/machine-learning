#!/bin/sh
set -x # turn on echoing of executed commands
set -e

docker build -t asurionpss/preprocessor-service .
docker run -t \
    -v ${PWD}:/preprocessor-service \
    -w /preprocessor-service \
    -p 5000:5000 \
    asurionpss/preprocessor-service \
    sh scripts/start.sh
