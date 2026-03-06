#pragma once
#include <complex>
#include <numbers>
#include "global.hpp"
#include "pluginshared/dsp/delay_line_multiple.hpp"
#include "pluginshared/dsp/delay_line_single.hpp"
#include "pluginshared/dsp/one_pole_tpt.hpp"
#include "pluginshared/simd.hpp"

namespace dsp {

// ----------------------------------------
// param
// ----------------------------------------

struct Param {
    float chorus_amount{0.05f};   // [0, 1]
    float chorus_freq{0.25f};     // [0.003, 8.0]
    float wet{0.25f};             // [0, 1]
    float pre_lowpass{0.0f};      // [0, 130]
    float pre_highpass{110.0f};   // [0, 130]
    float low_damp_pitch{0.0f};   // [0, 130]
    float high_damp_pitch{90.0f}; // [0, 130]
    float low_damp_db{0.0f};      // [-6, 0]
    float high_damp_db{-1.0f};    // [-6, 0]
    float size{0.5f};             // [0, 1]
    float decay_ms{1000.0f};      // [15ms, 64s]
    float pre_delay{0.0f};        // [0, 300ms]
};

// ----------------------------------------
// state
// ----------------------------------------

static constexpr float kT60Amplitude = 0.001f;
static constexpr float kAllpassFeedback = 0.6f;
static constexpr float kMinDelay = 3.0f;

static constexpr int kBaseSampleRate = 44100;
static constexpr int kDefaultSampleRate = 88200;
static constexpr int kNetworkSize = 16;
static constexpr int kBaseFeedbackBits = 14;
static constexpr int kExtraLookupSample = 4;
static constexpr int kBaseAllpassBits = 10;
static constexpr int kMinSizePower = -3;
static constexpr int kMaxSizePower = 1;
static constexpr float kSizePowerRange = kMaxSizePower - kMinSizePower;
template <class SimdType>
static constexpr int kNetworkContainers = kNetworkSize / simd::LaneSize<SimdType>;

static constexpr float kMaxChorusDrift = 2500.0f;
static constexpr float kMinDecayTime = 0.1f;
static constexpr float kMaxDecayTime = 100.0f;
static constexpr float kMaxChorusFrequency = 16.0f;
static constexpr float kChorusShiftAmount = 0.9f;
static constexpr float kSampleDelayMultiplier = 0.05f;
static constexpr float kSampleIncrementMultiplier = 0.05f;

static constexpr simd::Array256 kAllpassDelays{
    std::array{1001, 799, 933, 876, 895, 807, 907, 853, 957, 1019, 711, 567, 833, 779, 663, 997}
};
static constexpr simd::Array256 kFeedbackDelays{
    std::array<float, kNetworkSize>{6753.2f, 9278.4f, 7704.5f, 11328.5f, 9701.12f, 5512.5f, 8480.45f, 5638.65f,
                                    3120.73f, 3429.5f, 3626.37f, 7713.52f, 4521.54f, 6518.97f, 5265.56f, 5630.25}
};

template <simd::IsSimdFloat SimdT>
struct LaneNState {
    static constexpr int kContainerSize = kNetworkSize / simd::LaneSize<SimdT>;

    pluginshared::dsp::DelayLineSingle<simd::Float128> predelay_;

    simd::Array<std::vector<SimdT>, kContainerSize> allpass_lookups_{};

    std::vector<float> feedback_memorie_;
    std::array<float*, kNetworkSize> feedback_ptrs_{};
    simd::Array<SimdT, kContainerSize> feedback_offsets_{};

    simd::Array<SimdT, kContainerSize> decays_{};

    simd::Array<pluginshared::dsp::OnePoleTPT<SimdT>, kContainerSize> low_shelf_filters_;
    simd::Array<pluginshared::dsp::OnePoleTPT<SimdT>, kContainerSize> high_shelf_filters_;

    pluginshared::dsp::OnePoleTPT<simd::Float128> low_pre_filter_;
    pluginshared::dsp::OnePoleTPT<simd::Float128> high_pre_filter_;

    float low_pre_coefficient_{};
    float high_pre_coefficient_{};
    float low_coefficient_{};
    float low_amplitude_{};
    float high_coefficient_{};
    float high_amplitude_{};
    float feedback_offset_smooth_factor_{};

    float chorus_phase_{};
    SimdT chorus_amount_{};
    float sample_delay_{};
    float sample_delay_increment_{};
    float dry_{};
    float wet_{};
    int write_index_{};

    int max_allpass_size_{};
    int max_feedback_size_{};
    int feedback_mask_{};
    int poly_allpass_mask_{};
    int allpass_write_pos_{};

    float fs_{};
    float fs_ratio_{};
    float buffer_scale_ratio_{};

    void WarpBuffer() noexcept {
        for (auto ptr : feedback_ptrs_) {
            ptr[max_feedback_size_] = ptr[0];
            ptr[max_feedback_size_ + 1] = ptr[1];
            ptr[max_feedback_size_ + 2] = ptr[2];
            ptr[max_feedback_size_ + 3] = ptr[3];
        }
    }

    simd::Float128 ReadFeedback(size_t idx, simd::Float128 offset) noexcept {
        simd::Float128 rpos = (static_cast<float>(write_index_ + feedback_mask_)) - offset;
        auto irpos = (simd::ToInt128(rpos) - 1) & feedback_mask_;
        simd::Float128 t = simd::Frac128(rpos);

        // load [-1, 0, 1, 2]
        auto [yn1, y0, y1, y2] = simd::Transpose(simd::Loadu128(feedback_ptrs_[idx * 4] + irpos[0]),
                                                 simd::Loadu128(feedback_ptrs_[idx * 4 + 1] + irpos[1]),
                                                 simd::Loadu128(feedback_ptrs_[idx * 4 + 2] + irpos[2]),
                                                 simd::Loadu128(feedback_ptrs_[idx * 4 + 3] + irpos[3]));

        auto d0 = (y1 - yn1) * (0.5f);
        auto d1 = (y2 - y0) * (0.5f);
        auto d = y1 - y0;
        auto m0 = (3.0f) * d - (2.0f) * d0 - d1;
        auto m1 = d0 - (2.0f) * d + d1;
        return y0 + t * (d0 + t * (m0 + t * m1));
    }

    simd::Float256 ReadFeedback(size_t idx, simd::Float256 offset) noexcept {
        simd::Float256 rpos = (static_cast<float>(write_index_ + feedback_mask_)) - offset;
        auto irpos = (simd::ToInt256(rpos) - 1) & feedback_mask_;
        simd::Float256 t = simd::Frac256(rpos);

        // load [-1, 0, 1, 2]
        auto [yn1, y0, y1, y2] = simd::Transpose256(simd::Loadu128(feedback_ptrs_[idx * 8] + irpos[0]),
                                                    simd::Loadu128(feedback_ptrs_[idx * 8 + 1] + irpos[1]),
                                                    simd::Loadu128(feedback_ptrs_[idx * 8 + 2] + irpos[2]),
                                                    simd::Loadu128(feedback_ptrs_[idx * 8 + 3] + irpos[3]),
                                                    simd::Loadu128(feedback_ptrs_[idx * 8 + 4] + irpos[4]),
                                                    simd::Loadu128(feedback_ptrs_[idx * 8 + 5] + irpos[5]),
                                                    simd::Loadu128(feedback_ptrs_[idx * 8 + 6] + irpos[6]),
                                                    simd::Loadu128(feedback_ptrs_[idx * 8 + 7] + irpos[7]));

        auto d0 = (y1 - yn1) * (0.5f);
        auto d1 = (y2 - y0) * (0.5f);
        auto d = y1 - y0;
        auto m0 = (3.0f) * d - (2.0f) * d0 - d1;
        auto m1 = d0 - (2.0f) * d + d1;
        return y0 + t * (d0 + t * (m0 + t * m1));
    }

    simd::Float128 ReadAllpass(size_t i, simd::Int128 offset) noexcept {
        auto& buffer = allpass_lookups_[i];
        auto irpos = (allpass_write_pos_ + poly_allpass_mask_) - offset;
        irpos &= (poly_allpass_mask_);
        auto const* raw = reinterpret_cast<const float*>(buffer.data());
        return simd::Float128{raw[static_cast<size_t>(irpos[0]) * 4 + 0], raw[static_cast<size_t>(irpos[1]) * 4 + 1],
                              raw[static_cast<size_t>(irpos[2]) * 4 + 2], raw[static_cast<size_t>(irpos[3]) * 4 + 3]};
    }

    simd::Float256 ReadAllpass(size_t i, simd::Int256 offset) noexcept {
        auto& buffer = allpass_lookups_[i];
        auto irpos = (allpass_write_pos_ + poly_allpass_mask_) - offset;
        irpos &= (poly_allpass_mask_);
#ifndef SIMDE_X86_AVX2_NATIVE
        auto const* raw = reinterpret_cast<const float*>(buffer.data());
        return simd::Float256{raw[static_cast<size_t>(irpos[0]) * 8 + 0], raw[static_cast<size_t>(irpos[1]) * 8 + 1],
                              raw[static_cast<size_t>(irpos[2]) * 8 + 2], raw[static_cast<size_t>(irpos[3]) * 8 + 3],
                              raw[static_cast<size_t>(irpos[4]) * 8 + 4], raw[static_cast<size_t>(irpos[5]) * 8 + 5],
                              raw[static_cast<size_t>(irpos[6]) * 8 + 6], raw[static_cast<size_t>(irpos[7]) * 8 + 7]};
#else
        static const int32_t lane_offsets_data[8] = {0, 1, 2, 3, 4, 5, 6, 7};
        simde__m256i lane_offsets = simde_mm256_loadu_si256(lane_offsets_data);
        
        // vindex = irpos << 3 (即 * 8) + lane_offsets
        simde__m256i vindex = simde_mm256_add_epi32(simde_mm256_slli_epi32(irpos, 3), lane_offsets);

        // 3. 执行 Gather
        // scale = 4，因为我们读的是 float (4 bytes)
        float const* raw = reinterpret_cast<const float*>(buffer.data());
        return simde_mm256_i32gather_ps(raw, vindex, 4);
#endif
    }
};

struct ProcessorState {
    Param param;

    LaneNState<simd::Float128> lane4;
    LaneNState<simd::Float256> lane8;
};

struct ProcessorDsp {
    void (*init)(ProcessorState& state, float fs) noexcept;
    void (*reset)(ProcessorState& state) noexcept;
    void (*update)(ProcessorState& state, const Param& p) noexcept;
    void (*process)(ProcessorState& state, float* left, float* right, int num_samples) noexcept;

    bool IsValid() const noexcept {
        return init != nullptr && reset != nullptr && update != nullptr && process != nullptr;
    }
};

ProcessorDsp GetProcessorDsp() noexcept;
} // namespace dsp
