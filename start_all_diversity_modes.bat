@echo off
REM Generate games with ALL diversity modes for maximum training data variety
echo ========================================
echo KataHex - ALL Diversity Modes
echo ========================================
echo.
echo This will run multiple diversity modes sequentially:
echo   1. Strength diversity (weak vs strong)
echo   2. Temperature diversity (aggressive vs conservative)
echo   3. Exploration diversity (high vs low exploration)
echo   4. Speed diversity (fast vs slow)
echo   5. Random diversity (randomized parameters)
echo.
echo Total: 2500 games will be generated
echo.
pause

echo.
echo [1/5] Running STRENGTH diversity mode...
python generate_rollouts.py --diversity-mode strength --num-games 500 --num-threads 4

echo.
echo [2/5] Running TEMPERATURE diversity mode...
python generate_rollouts.py --diversity-mode temperature --num-games 500 --num-threads 4

echo.
echo [3/5] Running EXPLORATION diversity mode...
python generate_rollouts.py --diversity-mode exploration --num-games 500 --num-threads 4

echo.
echo [4/5] Running SPEED diversity mode...
python generate_rollouts.py --diversity-mode speed --num-games 500 --num-threads 4

echo.
echo [5/5] Running RANDOM diversity mode...
python generate_rollouts.py --diversity-mode random --num-games 500 --num-threads 4

echo.
echo ========================================
echo ALL DIVERSITY MODES COMPLETE!
echo ========================================
echo Total games generated: 2500
echo Check rollouts_output directory for results
echo.
pause

