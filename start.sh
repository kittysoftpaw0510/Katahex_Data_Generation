#!/bin/bash
# KataHex Rollout Generation - Diversity Mode
# This script generates diverse training data using different bot configurations

echo ========================================
echo KataHex Diverse Rollout Generation
echo ========================================
echo.
echo This will generate games with varied playing styles for rich training data.
echo.

# Run with random diversity mode for maximum variety
python generate_rollouts.py --diversity-mode random --num-games 100000 --num-threads 30 --num-gpus 1

echo.
echo ========================================
echo Batch complete!
echo ========================================
