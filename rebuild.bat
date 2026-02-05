@echo off
echo ========================================
echo Rebuilding KataHex with Fixed Connection Logic
echo ========================================
echo.

cd build

echo Cleaning old build files...
del /Q CMakeCache.txt 2>nul
rmdir /S /Q CMakeFiles 2>nul

echo.
echo Configuring with CMake...
cmake ../cpp -DUSE_BACKEND=EIGEN -DCOMPILE_MAX_BOARD_LEN=11 -DNO_GIT_REVISION=1 -DEIGEN3_INCLUDE_DIRS=E:/BitTensor/mini/eigen-5.0.0 -DZLIB_INCLUDE_DIR=E:/BitTensor/mini/zlib-1.2.13 -DZLIB_LIBRARY=E:/BitTensor/mini/zlib/static_x64/zlibstat.lib

echo.
echo Building...
cmake --build . --config Release

echo.
echo ========================================
echo Build complete!
echo ========================================
echo Executable: build\katahex-win64-19-eigen.exe
echo.
pause

