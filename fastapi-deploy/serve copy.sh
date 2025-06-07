#!/bin/bash

export PYTHONPATH=/app

if [ "$1" == "serve" ]; then
  # Start your FastAPI app with uvicorn on port 8080 and host 0.0.0.0
  uvicorn app.main:app --host 0.0.0.0 --port 8080
else
  # Fallback to execute any other commands if passed
  exec "$@"
fi
