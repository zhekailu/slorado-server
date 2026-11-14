# W8A8 quantized FAST v5.2 model: QuantCRFModel and the fused CUDA kernels it calls.
# Included from the top-level Makefile after the cuda/rocm block and before the link rules,
# so everything this fork adds to the build lives here.

OBJ        += $(BUILD_DIR)/QuantCRFModel.o
SERVER_OBJ += $(BUILD_DIR)/QuantCRFModel.o

$(BUILD_DIR)/QuantCRFModel.o: thirdparty/dorado/QuantCRFModel.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(DEPFLAGS) $< -c -o $@

ifdef cuda
NVCC      ?= $(CUDA_ROOT)/bin/nvcc
# mma.sync m16n8k32 s8 needs sm_80+.  An explicit CUDA_ARCH wins; otherwise build for the GPU in
# this machine (-arch=native, nvcc >= 11.5 with a visible GPU), else compute_80 PTX, which the
# driver JIT-compiles on any newer GPU.
#
# w8a8_release=1 builds a binary for distribution instead: SASS for every sm_80+ family this
# nvcc knows (Ampere, Orin, Ada, Hopper, Blackwell) plus compute_80 PTX, which the driver
# JIT-compiles on any GPU newer than the list.
comma := ,
W8A8_RELEASE_SMS := 80 86 87 89 90 100 120
ifeq ($(origin W8A8_ARCH),undefined)
ifneq ($(w8a8_release),)
W8A8_NVCC_ARCHS := $(shell $(NVCC) --list-gpu-arch 2>/dev/null)
W8A8_ARCH := $(foreach s,$(W8A8_RELEASE_SMS),$(if $(filter compute_$(s),$(W8A8_NVCC_ARCHS)),-gencode arch=compute_$(s)$(comma)code=sm_$(s))) \
             -gencode arch=compute_80,code=compute_80
else ifneq ($(CUDA_ARCH),)
W8A8_ARCH := $(CUDA_ARCH)
else ifeq ($(shell $(NVCC) -arch=native -x cu -c /dev/null -o /dev/null >/dev/null 2>&1 && echo ok),ok)
W8A8_ARCH := -arch=native
else
W8A8_ARCH := -gencode arch=compute_80,code=compute_80
endif
endif
W8A8_OBJ   = $(BUILD_DIR)/w8a8_fast.o $(BUILD_DIR)/w8a8_tanh_approx.o

OBJ        += $(W8A8_OBJ)
SERVER_OBJ += $(W8A8_OBJ)

$(BUILD_DIR)/w8a8_%.o: thirdparty/dorado/w8a8_%.cu thirdparty/dorado/w8a8_fast.h
	$(NVCC) -O3 -lineinfo $(W8A8_ARCH) -Xcompiler -Wall -c $< -o $@
endif
