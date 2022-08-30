#!/usr/bin/bash
ENV=".venv"
python3 -m venv $ENV
. $ENV/bin/activate && pip install --upgrade pip && pip install -r requirements-dev.txt
