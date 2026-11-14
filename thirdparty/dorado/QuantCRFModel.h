#ifndef QUANT_CRF_MODEL_H
#define QUANT_CRF_MODEL_H

#include <torch/torch.h>

#include <string>
#include <vector>

#include "model_config.h"
#include "tensor_chunk_utils.h"

#ifdef HAVE_CUDA
#include "w8a8_fast.h"
#endif

using namespace torch::nn;

// W8A8 FAST v5.2 models (quantization.json "format": "slorado-w8a8-gpu-v1"), run with the fused
// kernels in w8a8_fast.cu.  The model directory keeps the FP model's config.toml, so the config is
// loaded with load_lstm_model_config as usual; only the weights and the forward pass differ.

// true if model_path/quantization.json declares "format": "slorado-w8a8-gpu-v1"
bool is_w8a8_model(const std::string &model_path);

ModuleHolder<AnyModule> load_quant_lstm_model(const CRFModelConfig &model_config, const torch::TensorOptions &options, lstm_stats_t *model_stats);

struct QuantCRFModelImpl : torch::nn::Module {
    QuantCRFModelImpl(const CRFModelConfig &config, const torch::Device &device, lstm_stats_t *model_stats);

    // Input x is [N, 1, T] FP16 on the GPU.  Output is [N, T / stride, 256] FP16, a transposed
    // view of a contiguous [T', N, C] tensor (the TNC layout the decoder wants).
    torch::Tensor forward(const torch::Tensor &x);

private:
    // Packed device images of the weights.  Plain members rather than registered parameters or
    // buffers, so module->to(dtype) can never convert the INT8 images.
    torch::Tensor conv0_w_, conv0_dq_, conv0_b_;
    torch::Tensor conv1_frag_, conv1_dq_, conv1_b_;
    torch::Tensor conv2_frag_, conv2_dq_, conv2_b_;
    std::vector<torch::Tensor> lstm_frag_, lstm_epi_;
    torch::Tensor linear_frag_, linear_dq_;
#ifdef HAVE_CUDA
    w8a8f_frontend_params front_;  // points into the conv tensors above
#endif
    float inv_sh_ = 1.0f, clamp_lo_ = 0.0f, clamp_hi_ = 0.0f;
    int device_idx_ = 0;
    void *events_[4] = {};         // cudaEvent_t: stage boundaries for model_stats_ timing
    lstm_stats_t *model_stats_;
};

TORCH_MODULE(QuantCRFModel);

#endif
