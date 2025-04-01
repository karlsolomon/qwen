#!/bin/bash
source test/bin/activate
pip install -r requirements.txt
python -m uvicorn server:app --host 0.0.0.0 --port 11434
