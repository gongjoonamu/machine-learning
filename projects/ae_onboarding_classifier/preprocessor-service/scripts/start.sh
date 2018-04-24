#!/bin/sh
set -x # turn on echoing of executed commands
set -e

pip install Flask
FLASK_APP=application.py flask run
