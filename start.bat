@echo off
REM KataHex Rollout Generation - Diversity Mode
REM This script generates diverse training data using different bot configurations

echo ========================================
echo KataHex Diverse Rollout Generation
echo ========================================
echo.
echo This will generate games with varied playing styles for rich training data.
echo.

REM Run with random diversity mode for maximum variety
python generate_rollouts.py --diversity-mode random --num-games 40 --num-threads 4

echo.
echo ========================================
echo Batch complete!
echo ========================================
pause