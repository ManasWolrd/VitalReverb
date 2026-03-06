#pragma once
#include <complex>
#include <numbers>
#include "global.hpp"
#include "pluginshared/dsp/delay_line_multiple.hpp"
#include "pluginshared/dsp/delay_line_single.hpp"
#include "pluginshared/dsp/one_pole_tpt.hpp"
#include "pluginshared/simd.hpp"
#include "pluginshared/align_allocator.hpp"

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

    std::vector<float, simd::AlignedAllocator<float, 32>> feedback_memorie_;
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
#ifndef SIMDE_X86_AVX2_NATIVE
        for (auto ptr : feedback_ptrs_) {
            ptr[max_feedback_size_] = ptr[0];
            ptr[max_feedback_size_ + 1] = ptr[1];
            ptr[max_feedback_size_ + 2] = ptr[2];
            ptr[max_feedback_size_ + 3] = ptr[3];
        }
#else
        size_t raw_offset = max_feedback_size_ * kNetworkSize;
        float* dst = feedback_memorie_.data() + raw_offset;
        float* src = feedback_memorie_.data();
        for (int i = 0; i < kNetworkSize / 8 * 4; ++i) {
            simde_mm256_store_ps(dst, simde_mm256_load_ps(src));
            src += 8;
            dst += 8;
        }
#endif
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

#ifndef SIMDE_X86_AVX2_NATIVE
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
#else
        simde__m256i lane_ids = simde_mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        simde__m256i base_vindex = simde_mm256_add_epi32(
            simde_mm256_slli_epi32(simd::ToSimde(irpos), 4), simde_mm256_add_epi32(simde_mm256_set1_epi32(idx * 8), lane_ids));

        float const* raw = feedback_memorie_.data();
        auto yn1 = simd::FromSimde(simde_mm256_i32gather_ps(raw, base_vindex, 4));
        auto y0 = simd::FromSimde(
            simde_mm256_i32gather_ps(raw, simde_mm256_add_epi32(base_vindex, simde_mm256_set1_epi32(16)), 4));
        auto y1 = simd::FromSimde(
            simde_mm256_i32gather_ps(raw, simde_mm256_add_epi32(base_vindex, simde_mm256_set1_epi32(32)), 4));
        auto y2 = simd::FromSimde(
            simde_mm256_i32gather_ps(raw, simde_mm256_add_epi32(base_vindex, simde_mm256_set1_epi32(48)), 4));

        // simd::Float256 yn1;
        // yn1[0] = feedback_memorie_[irpos[0] * 16 + idx * 8 + 0];
        // yn1[1] = feedback_memorie_[irpos[1] * 16 + idx * 8 + 1];
        // yn1[2] = feedback_memorie_[irpos[2] * 16 + idx * 8 + 2];
        // yn1[3] = feedback_memorie_[irpos[3] * 16 + idx * 8 + 3];
        // yn1[4] = feedback_memorie_[irpos[4] * 16 + idx * 8 + 4];
        // yn1[5] = feedback_memorie_[irpos[5] * 16 + idx * 8 + 5];
        // yn1[6] = feedback_memorie_[irpos[6] * 16 + idx * 8 + 6];
        // yn1[7] = feedback_memorie_[irpos[7] * 16 + idx * 8 + 7];

        // simd::Float256 y0;
        // y0[0] = feedback_memorie_[irpos[0] * 16 + 16 + idx * 8 + 0];
        // y0[1] = feedback_memorie_[irpos[1] * 16 + 16 + idx * 8 + 1];
        // y0[2] = feedback_memorie_[irpos[2] * 16 + 16 + idx * 8 + 2];
        // y0[3] = feedback_memorie_[irpos[3] * 16 + 16 + idx * 8 + 3];
        // y0[4] = feedback_memorie_[irpos[4] * 16 + 16 + idx * 8 + 4];
        // y0[5] = feedback_memorie_[irpos[5] * 16 + 16 + idx * 8 + 5];
        // y0[6] = feedback_memorie_[irpos[6] * 16 + 16 + idx * 8 + 6];
        // y0[7] = feedback_memorie_[irpos[7] * 16 + 16 + idx * 8 + 7];

        // simd::Float256 y1;
        // y1[0] = feedback_memorie_[irpos[0] * 16 + 32 + idx * 8 + 0];
        // y1[1] = feedback_memorie_[irpos[1] * 16 + 32 + idx * 8 + 1];
        // y1[2] = feedback_memorie_[irpos[2] * 16 + 32 + idx * 8 + 2];
        // y1[3] = feedback_memorie_[irpos[3] * 16 + 32 + idx * 8 + 3];
        // y1[4] = feedback_memorie_[irpos[4] * 16 + 32 + idx * 8 + 4];
        // y1[5] = feedback_memorie_[irpos[5] * 16 + 32 + idx * 8 + 5];
        // y1[6] = feedback_memorie_[irpos[6] * 16 + 32 + idx * 8 + 6];
        // y1[7] = feedback_memorie_[irpos[7] * 16 + 32 + idx * 8 + 7];

        // simd::Float256 y2;
        // y2[0] = feedback_memorie_[irpos[0] * 16 + 48 + idx * 8 + 0];
        // y2[1] = feedback_memorie_[irpos[1] * 16 + 48 + idx * 8 + 1];
        // y2[2] = feedback_memorie_[irpos[2] * 16 + 48 + idx * 8 + 2];
        // y2[3] = feedback_memorie_[irpos[3] * 16 + 48 + idx * 8 + 3];
        // y2[4] = feedback_memorie_[irpos[4] * 16 + 48 + idx * 8 + 4];
        // y2[5] = feedback_memorie_[irpos[5] * 16 + 48 + idx * 8 + 5];
        // y2[6] = feedback_memorie_[irpos[6] * 16 + 48 + idx * 8 + 6];
        // y2[7] = feedback_memorie_[irpos[7] * 16 + 48 + idx * 8 + 7];

        auto d0 = (y1 - yn1) * (0.5f);
        auto d1 = (y2 - y0) * (0.5f);
        auto d = y1 - y0;
        auto m0 = (3.0f) * d - (2.0f) * d0 - d1;
        auto m1 = d0 - (2.0f) * d + d1;
        return y0 + t * (d0 + t * (m0 + t * m1));
#endif
    }

    void PushFeedback(simd::Float256 store1, simd::Float256 store2) noexcept {
#ifndef SIMDE_X86_AVX2_NATIVE
        int const write_idx = write_index_;
        auto* const ptrs = feedback_ptrs_.data();
        ptrs[0][write_idx] = store1[0];
        ptrs[1][write_idx] = store1[1];
        ptrs[2][write_idx] = store1[2];
        ptrs[3][write_idx] = store1[3];
        ptrs[4][write_idx] = store1[4];
        ptrs[5][write_idx] = store1[5];
        ptrs[6][write_idx] = store1[6];
        ptrs[7][write_idx] = store1[7];
        ptrs[8][write_idx] = store2[0];
        ptrs[9][write_idx] = store2[1];
        ptrs[10][write_idx] = store2[2];
        ptrs[11][write_idx] = store2[3];
        ptrs[12][write_idx] = store2[4];
        ptrs[13][write_idx] = store2[5];
        ptrs[14][write_idx] = store2[6];
        ptrs[15][write_idx] = store2[7];
#else
        size_t offset = write_index_ * 16;
        float* ptr = feedback_memorie_.data() + offset;
        simde_mm256_store_ps(ptr, simd::ToSimde(store1));
        simde_mm256_store_ps(ptr + 8, simd::ToSimde(store2));
        // feedback_memorie_[write_index_ * 16 + 0] = store1[0];
        // feedback_memorie_[write_index_ * 16 + 1] = store1[1];
        // feedback_memorie_[write_index_ * 16 + 2] = store1[2];
        // feedback_memorie_[write_index_ * 16 + 3] = store1[3];
        // feedback_memorie_[write_index_ * 16 + 4] = store1[4];
        // feedback_memorie_[write_index_ * 16 + 5] = store1[5];
        // feedback_memorie_[write_index_ * 16 + 6] = store1[6];
        // feedback_memorie_[write_index_ * 16 + 7] = store1[7];
        // feedback_memorie_[write_index_ * 16 + 8] = store2[0];
        // feedback_memorie_[write_index_ * 16 + 9] = store2[1];
        // feedback_memorie_[write_index_ * 16 + 10] = store2[2];
        // feedback_memorie_[write_index_ * 16 + 11] = store2[3];
        // feedback_memorie_[write_index_ * 16 + 12] = store2[4];
        // feedback_memorie_[write_index_ * 16 + 13] = store2[5];
        // feedback_memorie_[write_index_ * 16 + 14] = store2[6];
        // feedback_memorie_[write_index_ * 16 + 15] = store2[7];
#endif
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
        simde__m256i vindex = simde_mm256_add_epi32(simde_mm256_slli_epi32(simd::ToSimde(irpos), 3), lane_offsets);
        float const* raw = reinterpret_cast<const float*>(buffer.data());
        return simd::FromSimde(simde_mm256_i32gather_ps(raw, vindex, 4));
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
