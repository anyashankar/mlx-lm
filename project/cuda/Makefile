# Build on a GPU machine (GHC, or Bridges-2 after `module load cuda`).
# ARCH=native needs a visible GPU; otherwise pass e.g. ARCH=sm_75 (RTX 20xx), sm_70 (V100), sm_90 (H100).
NVCC ?= nvcc
ARCH ?= native
FLAGS = -O2 -std=c++17 -lineinfo

all: litmus replay

litmus: litmus.cu
	$(NVCC) $(FLAGS) -arch=$(ARCH) litmus.cu -o litmus

replay: replay.cu npy.h
	$(NVCC) $(FLAGS) -arch=$(ARCH) replay.cu -o replay

# CPU-only build of the replay (no CUDA needed): same math, serial loops.
replay_cpu: replay.cu npy.h
	g++ -O2 -std=c++17 -Wno-unknown-pragmas -x c++ replay.cu -o replay_cpu

clean:
	rm -f litmus replay replay_cpu

.PHONY: all clean
