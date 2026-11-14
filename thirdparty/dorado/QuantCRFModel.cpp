#include <errno.h>
#include <stdlib.h>
#include <string.h>

#include <fstream>
#include <sstream>
#include <string>

#include "QuantCRFModel.h"
#include "error.h"
#include "misc.h"

#ifdef HAVE_CUDA
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#endif

using namespace torch::nn;

static const char *W8A8_FORMAT = "slorado-w8a8-gpu-v1";

// ── quantization.json: flat keys, all unique, so a key lookup is all that is needed ──

static bool read_file(const std::string &path, std::string &out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::ostringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}

// Position of the value following "key":, or npos.
static size_t json_value_pos(const std::string &s, const std::string &key) {
    size_t pos = s.find("\"" + key + "\"");
    if (pos == std::string::npos) return pos;
    pos += key.size() + 2;
    while (pos < s.size() && isspace((unsigned char)s[pos])) ++pos;
    if (pos >= s.size() || s[pos] != ':') return std::string::npos;
    ++pos;
    while (pos < s.size() && isspace((unsigned char)s[pos])) ++pos;
    return pos;
}

static bool json_string(const std::string &s, const std::string &key, std::string &out) {
    size_t pos = json_value_pos(s, key);
    if (pos == std::string::npos || s[pos] != '"') return false;
    size_t end = s.find('"', pos + 1);
    if (end == std::string::npos) return false;
    out = s.substr(pos + 1, end - pos - 1);
    return true;
}

bool is_w8a8_model(const std::string &model_path) {
    std::string json, format;
    if (!read_file(model_path + "/quantization.json", json)) return false;
    return json_string(json, "format", format) && format == W8A8_FORMAT;
}

ModuleHolder<AnyModule> load_quant_lstm_model(const CRFModelConfig &model_config, const torch::TensorOptions &options, lstm_stats_t *model_stats) {
#ifndef HAVE_CUDA
    ERROR("W8A8 model %s needs the CUDA kernels in w8a8_fast.cu; rebuild slorado with cuda=1", model_config.model_path.c_str());
    exit(EXIT_FAILURE);
#endif
    if (!options.device().is_cuda()) {
        ERROR("W8A8 model %s runs only on a CUDA GPU (sm_80+); use -x cuda:N", model_config.model_path.c_str());
        exit(EXIT_FAILURE);
    }
    if (options.dtype().toScalarType() != torch::kHalf) {
        ERROR("%s", "W8A8 model expects FP16 input signal");
        exit(EXIT_FAILURE);
    }
    INFO("W8A8 model detected, using fused kernels on %s", options.device().str().c_str());

    auto model = QuantCRFModel(model_config, options.device(), model_stats);
    model->eval();

    auto module = AnyModule(model);
    auto holder = ModuleHolder<AnyModule>(module);
    return holder;
}

#ifdef HAVE_CUDA

// ── weights: plain torch.save files, which torch::load (TorchScript archives) cannot read ──

static torch::Tensor load_w8a8_tensor(const std::string &dir, const std::string &name, torch::ScalarType dtype, int64_t numel) {
    const auto path = dir + "/" + name + ".tensor";
    std::string bytes;
    if (!read_file(path, bytes)) {
        ERROR("cannot open %s: %s", path.c_str(), strerror(errno));
        exit(EXIT_FAILURE);
    }
    auto t = torch::pickle_load(std::vector<char>(bytes.begin(), bytes.end())).toTensor().contiguous();
    if (t.numel() != numel) {
        ERROR("%s: expected %ld elements, found %ld", path.c_str(), (long)numel, (long)t.numel());
        exit(EXIT_FAILURE);
    }
    if (dtype == torch::kInt8) {
        if (t.scalar_type() != torch::kInt8) {
            ERROR("%s: expected int8 weights", path.c_str());
            exit(EXIT_FAILURE);
        }
        return t;
    }
    // scales are FP16 on disk ("scale_dtype": "fp16"), biases FP32; FP16 -> FP32 is exact
    return t.to(torch::kFloat).contiguous();
}

static bool json_number(const std::string &s, const std::string &key, double &out) {
    size_t pos = json_value_pos(s, key);
    if (pos == std::string::npos) return false;
    char *end = NULL;
    out = strtod(s.c_str() + pos, &end);
    return end != s.c_str() + pos;
}

static float json_required(const std::string &json, const std::string &key) {
    double v;
    if (!json_number(json, key, v)) {
        ERROR("quantization.json: missing numeric field \"%s\"", key.c_str());
        exit(EXIT_FAILURE);
    }
    return (float)v;
}

static void json_expect(const std::string &json, const std::string &key, const char *want) {
    std::string v;
    if (!json_string(json, key, v) || v != want) {
        ERROR("quantization.json: \"%s\" is \"%s\", the W8A8 kernels require \"%s\"", key.c_str(), v.c_str(), want);
        exit(EXIT_FAILURE);
    }
}

static void check_rc(int rc, const char *what) {
    TORCH_CHECK(rc == 0, what, " failed: ", cudaGetErrorString((cudaError_t)rc));
}

QuantCRFModelImpl::QuantCRFModelImpl(const CRFModelConfig &config, const torch::Device &device, lstm_stats_t *model_stats)
        : model_stats_(model_stats) {
    const auto &dir = config.model_path;

    // The kernels hard-code the FAST v5.2 shapes
    const auto &cv = config.convs;
    const bool shapes_ok = cv.size() == 3 &&
        cv[0].insize == 1  && cv[0].size == 16 && cv[0].winlen == 5  && cv[0].stride == 1 &&
        cv[1].insize == 16 && cv[1].size == 16 && cv[1].winlen == 5  && cv[1].stride == 1 &&
        cv[2].insize == 16 && cv[2].size == 96 && cv[2].winlen == 19 && cv[2].stride == 6 &&
        config.lstm_size == 96 && config.lstm_layers == 5 && config.outsize == 256 && !config.has_out_features;
    if (!shapes_ok) {
        ERROR("%s: W8A8 kernels support only the FAST v5.2 architecture (convs 1-16-16-96, 5 x LSTM96, 256 outputs)", dir.c_str());
        exit(EXIT_FAILURE);
    }

    std::string json;
    if (!read_file(dir + "/quantization.json", json)) {
        ERROR("cannot open %s/quantization.json: %s", dir.c_str(), strerror(errno));
        exit(EXIT_FAILURE);
    }
    json_expect(json, "format", W8A8_FORMAT);
    json_expect(json, "elementwise", "fp32");
    json_expect(json, "gx_dtype", "fp16");
    json_expect(json, "nonlinearity", "tanh_approx");

    device_idx_ = device.has_index() ? device.index() : 0;
    cudaSetDevice(device_idx_);
    if (!w8a8f_supported()) {
        ERROR("%s: W8A8 kernels need an sm_80+ GPU", device.str().c_str());
        exit(EXIT_FAILURE);
    }

    // Activation quantization is x * (1/s) with 1/s rounded to FP32, as ATen computes it
    const float s_in = json_required(json, "conv0_in"), s_c1 = json_required(json, "conv1_in");
    const float s_c2 = json_required(json, "conv2_in"), s_x0 = json_required(json, "lstm0_x");
    const float s_h = json_required(json, "lstm_h");
    inv_sh_ = 1.0f / s_h;
    clamp_lo_ = json_required(json, "clamp_min");
    clamp_hi_ = json_required(json, "clamp_max");

    const auto u8 = torch::TensorOptions().dtype(torch::kUInt8);
    const auto f32 = torch::TensorOptions().dtype(torch::kFloat);

    // conv0 runs as scalar dot products: natural [16][1][5] weights
    conv0_w_  = load_w8a8_tensor(dir, "conv0.weight", torch::kInt8, 16 * 5).to(device);
    conv0_dq_ = load_w8a8_tensor(dir, "conv0.dequant_scale", torch::kFloat, 16).to(device);
    conv0_b_  = load_w8a8_tensor(dir, "conv0.bias", torch::kFloat, 16).to(device);

    auto w1 = load_w8a8_tensor(dir, "conv1.weight", torch::kInt8, 16 * 16 * 5);
    auto frag1 = torch::empty({W8A8F_CONV1_FRAG_BYTES}, u8);
    w8a8f_pack_conv1(w1.data_ptr<int8_t>(), frag1.data_ptr<uint8_t>());
    conv1_frag_ = frag1.to(device);
    conv1_dq_ = load_w8a8_tensor(dir, "conv1.dequant_scale", torch::kFloat, 16).to(device);
    conv1_b_  = load_w8a8_tensor(dir, "conv1.bias", torch::kFloat, 16).to(device);

    auto w2 = load_w8a8_tensor(dir, "conv2.weight", torch::kInt8, 96 * 16 * 19);
    auto frag2 = torch::empty({W8A8F_CONV2_FRAG_BYTES}, u8);
    w8a8f_pack_conv2(w2.data_ptr<int8_t>(), frag2.data_ptr<uint8_t>());
    conv2_frag_ = frag2.to(device);
    conv2_dq_ = load_w8a8_tensor(dir, "conv2.dequant_scale", torch::kFloat, 96).to(device);
    conv2_b_  = load_w8a8_tensor(dir, "conv2.bias", torch::kFloat, 96).to(device);

    for (int i = 0; i < config.lstm_layers; ++i) {
        const auto p = "lstm" + std::to_string(i) + ".";
        auto wih  = load_w8a8_tensor(dir, p + "wih", torch::kInt8, 384 * 96);
        auto whh  = load_w8a8_tensor(dir, p + "whh", torch::kInt8, 384 * 96);
        auto dqx  = load_w8a8_tensor(dir, p + "dequant_x", torch::kFloat, 384);
        auto dqh  = load_w8a8_tensor(dir, p + "dequant_h", torch::kFloat, 384);
        auto bias = load_w8a8_tensor(dir, p + "bias", torch::kFloat, 384);
        auto frag = torch::empty({W8A8F_LSTM_FRAG_BYTES}, u8);
        auto epi  = torch::empty({W8A8F_LSTM_EPI_FLOATS}, f32);
        w8a8f_pack_lstm(wih.data_ptr<int8_t>(), whh.data_ptr<int8_t>(), dqx.data_ptr<float>(),
                        dqh.data_ptr<float>(), bias.data_ptr<float>(), frag.data_ptr<uint8_t>(), epi.data_ptr<float>());
        lstm_frag_.push_back(frag.to(device));
        lstm_epi_.push_back(epi.to(device));
    }

    auto wl = load_w8a8_tensor(dir, "linear.weight", torch::kInt8, 256 * 96);
    auto fragl = torch::empty({W8A8F_LINEAR_FRAG_BYTES}, u8);
    w8a8f_pack_linear(wl.data_ptr<int8_t>(), fragl.data_ptr<uint8_t>());
    linear_frag_ = fragl.to(device);
    linear_dq_ = load_w8a8_tensor(dir, "linear.dequant_scale", torch::kFloat, 256).to(device);

    front_.w0 = conv0_w_.data_ptr<int8_t>();
    front_.dq0 = conv0_dq_.data_ptr<float>();
    front_.b0 = conv0_b_.data_ptr<float>();
    front_.w1frag = conv1_frag_.data_ptr<uint8_t>();
    front_.dq1 = conv1_dq_.data_ptr<float>();
    front_.b1 = conv1_b_.data_ptr<float>();
    front_.w2frag = conv2_frag_.data_ptr<uint8_t>();
    front_.dq2 = conv2_dq_.data_ptr<float>();
    front_.b2 = conv2_b_.data_ptr<float>();
    front_.inv_in = 1.0f / s_in;
    front_.inv_c1 = 1.0f / s_c1;
    front_.inv_c2 = 1.0f / s_c2;
    front_.inv_x0 = 1.0f / s_x0;
    front_.conv_clamp = json_required(json, "conv_clamp_max");

    for (auto &e : events_) check_rc(cudaEventCreate(reinterpret_cast<cudaEvent_t *>(&e)), "cudaEventCreate");
}

torch::Tensor QuantCRFModelImpl::forward(const torch::Tensor &x_in) {
    TORCH_CHECK(x_in.is_cuda() && x_in.scalar_type() == torch::kHalf && x_in.dim() == 3 && x_in.size(1) == 1,
                "QuantCRFModel expects [N, 1, T] FP16 CUDA input, got ", x_in.sizes(), " ",
                x_in.scalar_type(), " on ", x_in.device());
    const auto x = x_in.contiguous();
    const int N = x.size(0);
    const int T = x.size(2);
    const int Tp = (T - 1) / 6 + 1;

    // The kernels run on this binary's (static) CUDA runtime, whose current device is
    // independent of libtorch's DeviceGuard
    cudaSetDevice(device_idx_);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device_idx_).stream();
    cudaEvent_t ev[4];
    for (int i = 0; i < 4; ++i) ev[i] = reinterpret_cast<cudaEvent_t>(events_[i]);

    // INT8 [T'][N][96] activations, ping-ponged between layers
    auto act = torch::empty({Tp, N, 96}, x.options().dtype(torch::kInt8));
    auto act_next = torch::empty_like(act);
    // linear + clamp writes FP16 scores in TNC order
    auto scores = torch::empty({Tp, N, 256}, x.options());

    // Stage times come from events, so the stages run back to back with one wait at the end
    // (the caller synchronises after forward anyway).
    check_rc(cudaEventRecord(ev[0], stream), "cudaEventRecord");
    check_rc(w8a8f_frontend(x.data_ptr(), act.data_ptr<int8_t>(), N, T, &front_, stream), "w8a8f_frontend");
    check_rc(cudaEventRecord(ev[1], stream), "cudaEventRecord");
    for (size_t i = 0; i < lstm_frag_.size(); ++i) {
        // LSTMStack flips before every layer, so even layers run reversed
        const int reverse = (i & 1) == 0;
        check_rc(w8a8f_lstm(act.data_ptr<int8_t>(), act_next.data_ptr<int8_t>(), N, Tp, reverse,
                            lstm_frag_[i].data_ptr<uint8_t>(), lstm_epi_[i].data_ptr<float>(), inv_sh_, stream),
                 "w8a8f_lstm");
        std::swap(act, act_next);
    }
    check_rc(cudaEventRecord(ev[2], stream), "cudaEventRecord");
    check_rc(w8a8f_linear(act.data_ptr<int8_t>(), scores.data_ptr(), (int64_t)Tp * N, linear_frag_.data_ptr<uint8_t>(),
                          linear_dq_.data_ptr<float>(), clamp_lo_, clamp_hi_, stream),
             "w8a8f_linear");
    check_rc(cudaEventRecord(ev[3], stream), "cudaEventRecord");
    check_rc(cudaEventSynchronize(ev[3]), "cudaEventSynchronize");

    float ms[3];
    for (int i = 0; i < 3; ++i) check_rc(cudaEventElapsedTime(&ms[i], ev[i], ev[i + 1]), "cudaEventElapsedTime");
    model_stats_->time_conv_stack += ms[0] / 1000.0;
    model_stats_->time_rnns += ms[1] / 1000.0;
    model_stats_->time_crf_1 += ms[2] / 1000.0;

    // Output is [N, T, C], non-contiguous; the caller's transpose(0, 1).contiguous() is free
    return scores.transpose(0, 1);
}

#else

QuantCRFModelImpl::QuantCRFModelImpl(const CRFModelConfig &config, const torch::Device &, lstm_stats_t *model_stats)
        : model_stats_(model_stats) {
    ERROR("W8A8 model %s needs slorado built with cuda=1", config.model_path.c_str());
    exit(EXIT_FAILURE);
}

torch::Tensor QuantCRFModelImpl::forward(const torch::Tensor &x) {
    TORCH_CHECK(false, "QuantCRFModel needs slorado built with cuda=1");
    return x;
}

#endif
