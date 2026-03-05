#include "dsp_state.hpp"

#include <simd_detector.h>

namespace dsp {
ProcessorDsp GetProcessorDsp() noexcept {
    using IS = simd_detector::InstructionSet;

#if defined(__aarch64__) || defined(_M_ARM64)
    if (simd_detector::is_supported(IS::NEON)) {
        extern ProcessorDsp dsp_neon;
        return dsp_neon;
    }
#endif

#if defined(__x86_64__) || defined(_M_X64)
    if (simd_detector::is_supported(IS::AVX2)) {
        extern ProcessorDsp dsp_avx2;
        return dsp_avx2;
    }
    if (simd_detector::is_supported(IS::AVX)) {
        extern ProcessorDsp dsp_avx;
        return dsp_avx;
    }
    if (simd_detector::is_supported(IS::SSE4_1)) {
        extern ProcessorDsp dsp_sse4;
        return dsp_sse4;
    }
    if (simd_detector::is_supported(IS::SSE2)) {
        extern ProcessorDsp dsp_sse2;
        return dsp_sse2;
    }
#endif

    return ProcessorDsp{};
}
}
