#!/bin/bash
python -m spacy download en_core_web_sm
uvicorn app:app --host=0.0.0.0 --port=8080
