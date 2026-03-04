#include "dsp_state.hpp"

namespace dsp {

// ----------------------------------------
// dsp processor
// ----------------------------------------

static void Init(dsp::ProcessorState& state, float fs) noexcept {
}

static void Reset(dsp::ProcessorState& state) noexcept {

}

static void Update(dsp::ProcessorState& state, const dsp::Param& p) noexcept {
}

static void Process(dsp::ProcessorState& state, float* left, float* right, int num_samples) noexcept {
}

// ----------------------------------------
// export
// ----------------------------------------

#ifndef DSP_EXPORT_NAME
#error "不应该编译这个文件,在其他cpp包含这个cpp并定义DSP_EXPORT_NAME=`dsp_dispatch.cpp里的变量`"
#endif

ProcessorDsp DSP_EXPORT_NAME{
    Init,
    Reset,
    Update,
    Process
};
}
