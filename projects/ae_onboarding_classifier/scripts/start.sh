#!/bin/sh
set -x # turn on echoing of executed commands
set -e

pip install -r requirements.txt
FLASK_APP=application.py flask run --host=0.0.0.0
