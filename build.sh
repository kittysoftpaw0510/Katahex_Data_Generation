# Create build directory inside cpp
sudo apt update
sudo apt install zlib1g zlib1g-dev

mkdir -p cpp/build
cd cpp/build

# Now configure (.. points to cpp directory)
cmake .. \
    -DUSE_BACKEND=CUDA \
    -DNO_GIT_REVISION=1 \
    -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build . --config Release -j$(nproc)