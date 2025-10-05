#!/bin/bash
# Worker startup script with mandatory validation
# This ensures the worker never starts with invalid configuration

set -e  # Exit immediately if any command fails

echo "========================================================================"
echo "WORKER STARTUP - RUNNING MANDATORY VALIDATION"
echo "========================================================================"

# Run the validation script - will exit with error if validation fails
python /app/worker_startup_check.py

# If we get here, validation passed - start the worker
echo ""
echo "Validation passed! Starting worker..."
echo ""

# Start RQ worker
exec python -m rq.cli worker --url redis://redis:6379 video_generation
