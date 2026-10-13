# W8A8 quantized FAST v5.2 model: QuantCRFModel and the fused CUDA kernels it calls.
# Included from the top-level Makefile after the cuda/rocm block and before the link rules,
# so everything this fork adds to the build lives here.

OBJ        += $(BUILD_DIR)/QuantCRFModel.o
SERVER_OBJ += $(BUILD_DIR)/QuantCRFModel.o

$(BUILD_DIR)/QuantCRFModel.o: thirdparty/dorado/QuantCRFModel.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(DEPFLAGS) $< -c -o $@

ifdef cuda
NVCC      ?= $(CUDA_ROOT)/bin/nvcc
# mma.sync m16n8k32 s8 needs sm_80+; compute_80 PTX JIT-compiles on any newer GPU
W8A8_ARCH ?= $(if $(CUDA_ARCH),$(CUDA_ARCH),-gencode arch=compute_80,code=compute_80)
W8A8_OBJ   = $(BUILD_DIR)/w8a8_fast.o $(BUILD_DIR)/w8a8_tanh_approx.o

OBJ        += $(W8A8_OBJ)
SERVER_OBJ += $(W8A8_OBJ)

$(BUILD_DIR)/w8a8_%.o: thirdparty/dorado/w8a8_%.cu thirdparty/dorado/w8a8_fast.h
	$(NVCC) -O3 -lineinfo $(W8A8_ARCH) -Xcompiler -Wall -c $< -o $@
endif
