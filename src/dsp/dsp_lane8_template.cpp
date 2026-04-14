#include "dsp_state.hpp"

#include <qwqdsp/convert.hpp>

namespace dsp {

// ----------------------------------------
// dsp processor
// ----------------------------------------

static void Init(dsp::ProcessorState& state, float fs) noexcept {
    auto& self = state.lane8;

    self.fs_ = fs;
    self.fs_ratio_ = fs / kBaseSampleRate;

    self.predelay_.Init(fs, 300.0f);

    self.low_pre_coefficient_ = 0.1f;
    self.high_pre_coefficient_ = 0.1f;
    self.low_coefficient_ = 0.1f;
    self.high_coefficient_ = 0.1f;
    self.low_amplitude_ = 0.0f;
    self.high_amplitude_ = 0.0f;
    self.sample_delay_ = kMinDelay;

    float const buffer_scale_ratio = fs / kBaseSampleRate;
    self.buffer_scale_ratio_ = buffer_scale_ratio;
    uint32_t const base_feedback_size =
        static_cast<uint32_t>(std::ceil(buffer_scale_ratio * (1 << (kBaseFeedbackBits + kMaxSizePower))));
    uint32_t max_feedback_size = 1;
    while (max_feedback_size < base_feedback_size) {
        max_feedback_size *= 2;
    }
    self.max_feedback_size_ = static_cast<int>(max_feedback_size);
    self.feedback_mask_ = self.max_feedback_size_ - 1;

    uint32_t each_size = max_feedback_size + kExtraLookupSample;
    self.feedback_memorie_.resize(each_size * kNetworkSize);
    for (int i = 0; i < kNetworkSize; ++i) {
        self.feedback_ptrs_[static_cast<size_t>(i)] = &self.feedback_memorie_[static_cast<size_t>(i) * each_size];
    }

    uint32_t const base_allpass_size = static_cast<uint32_t>(std::ceil(buffer_scale_ratio * (1 << kBaseAllpassBits)));
    uint32_t max_allpass_size = 1;
    while (max_allpass_size < base_allpass_size) {
        max_allpass_size *= 2;
    }
    self.max_allpass_size_ = static_cast<int>(max_allpass_size);
    self.poly_allpass_mask_ = self.max_allpass_size_ - 1;
    for (auto& buffer : self.allpass_lookups_) {
        buffer.resize(static_cast<size_t>(max_allpass_size));
    }

    self.feedback_offset_smooth_factor_ = 1.0f - std::exp(-1.0f / (fs * 50.0f / 1000.0f));

    self.write_index_ = 0;
    self.allpass_write_pos_ = 0;
}

static void Reset(dsp::ProcessorState& state) noexcept {
    auto& self = state.lane8;
    const auto& param = state.param;

    self.wet_ = 0;
    self.dry_ = 0;
    self.chorus_amount_ = simd::BroadcastF256(param.chorus_amount * kMaxChorusDrift);

    for (auto& f : self.low_shelf_filters_) {
        f.Reset();
    }
    for (auto& f : self.high_shelf_filters_) {
        f.Reset();
    }
    self.low_pre_filter_.Reset();
    self.high_pre_filter_.Reset();
    self.predelay_.Reset();
    for (auto& d : self.decays_) {
        d = simd::BroadcastF256(0);
    }

    const simd::Float256* feedback_delays = (const simd::Float256*)kFeedbackDelays.data();
    for (size_t i = 0; i < self.kContainerSize; ++i) {
        self.feedback_offsets_[i] = feedback_delays[i];
    }

    for (auto& buffer : self.allpass_lookups_) {
        std::fill(buffer.begin(), buffer.end(), simd::Float256{});
    }
    std::fill(self.feedback_memorie_.begin(), self.feedback_memorie_.end(), 0.0f);
}

static void Update(dsp::ProcessorState& state, const dsp::Param& p) noexcept {
    state.param = p;
}

// Float256(Ao0, Ao1) -> Float256(ReduceAdd(Ao0), ReduceAdd(Ao1))
static inline simd::Float256 _InternalSum(simd::Float256 x) noexcept {
#ifdef SIMDE_X86_AVX2_NATIVE
    simde__m256 t1 = simde_mm256_add_ps(simd::ToSimde(x), simde_mm256_permute_ps(x, SIMDE_MM_SHUFFLE(2, 3, 0, 1)));
    simde__m256 y = simde_mm256_add_ps(t1, simde_mm256_permute_ps(t1, SIMDE_MM_SHUFFLE(1, 0, 3, 2)));
    return simd::FromSimde(y);
#else
    simde__m256 sum_v = simd::ToSimde(x);
    sum_v = simde_mm256_add_ps(sum_v, simde_mm256_shuffle_ps(sum_v, sum_v, SIMDE_MM_SHUFFLE(1, 0, 3, 2)));
    sum_v = simde_mm256_add_ps(sum_v, simde_mm256_shuffle_ps(sum_v, sum_v, SIMDE_MM_SHUFFLE(2, 3, 0, 1)));
    return simd::FromSimde(sum_v);
#endif
}

static inline void _ScatterLane8(simd::Float256 in1, simd::Float256 in2, simd::Float256& out1,
                                 simd::Float256& out2) noexcept {
    alignas(32) auto [ao1, ao2] = simd::Break(in1);
    alignas(32) auto [ao3, ao4] = simd::Break(in2);
    auto row_sum = ao1 + ao2 + ao3 + ao4;
    auto total_rows = simd::Combine(row_sum, row_sum);
    auto sum_all = simd::ReduceAdd(in1 + in2);
    auto sum_ao01 = _InternalSum(in1);
    auto sum_ao23 = _InternalSum(in2);
    auto common = 0.25f * sum_all;
    out1 = common + in1 - 0.5f * (sum_ao01 + total_rows);
    out2 = common + in2 - 0.5f * (sum_ao23 + total_rows);
}

template <bool kMono>
static void ProcessInternal(dsp::ProcessorState& state, float* left, float* right, int num_samples) noexcept {
    auto& self = state.lane8;
    const auto& param = state.param;

    self.WarpBuffer();

    float const tick_increment = 1.0f / static_cast<float>(num_samples);

    float current_dry = self.dry_;
    float current_wet = self.wet_;
    float current_low_pre_coefficient = self.low_pre_coefficient_;
    float current_high_pre_coefficient = self.high_pre_coefficient_;
    float current_low_coefficient = self.low_coefficient_;
    float current_low_amplitude = self.low_amplitude_;
    float current_high_coefficient = self.high_coefficient_;
    float current_high_amplitude = self.high_amplitude_;

    self.wet_ = std::sin(param.wet * std::numbers::pi_v<float> / 2);
    self.dry_ = std::cos(param.wet * std::numbers::pi_v<float> / 2);
    float delta_wet = (self.wet_ - current_wet) * tick_increment;
    float delta_dry = (self.dry_ - current_dry) * tick_increment;

    float const low_pre_cutoff_frequency = qwqdsp::convert::Pitch2Freq(param.pre_lowpass);
    self.low_pre_coefficient_ =
        decltype(self.low_pre_filter_)::ComputeCoeff(qwqdsp::convert::Freq2W(low_pre_cutoff_frequency, self.fs_));
    float const high_pre_cutoff_frequency = qwqdsp::convert::Pitch2Freq(param.pre_highpass);
    self.high_pre_coefficient_ =
        decltype(self.high_pre_filter_)::ComputeCoeff(qwqdsp::convert::Freq2W(high_pre_cutoff_frequency, self.fs_));
    float delta_low_pre_coefficient = (self.low_pre_coefficient_ - current_low_pre_coefficient) * tick_increment;
    float delta_high_pre_coefficient = (self.high_pre_coefficient_ - current_high_pre_coefficient) * tick_increment;

    float const low_cutoff_frequency = qwqdsp::convert::Pitch2Freq(param.low_damp_pitch);
    self.low_coefficient_ =
        decltype(self.low_pre_filter_)::ComputeCoeff(qwqdsp::convert::Freq2W(low_cutoff_frequency, self.fs_));
    float const high_cutoff_frequency = qwqdsp::convert::Pitch2Freq(param.high_damp_pitch);
    self.high_coefficient_ =
        decltype(self.high_pre_filter_)::ComputeCoeff(qwqdsp::convert::Freq2W(high_cutoff_frequency, self.fs_));
    float delta_low_coefficient = (self.low_coefficient_ - current_low_coefficient) * tick_increment;
    float delta_high_coefficient = (self.high_coefficient_ - current_high_coefficient) * tick_increment;

    self.low_amplitude_ = 1.0f - qwqdsp::convert::Db2Gain(param.low_damp_db);
    self.high_amplitude_ = qwqdsp::convert::Db2Gain(param.high_damp_db);
    float delta_low_amplitude = (self.low_amplitude_ - current_low_amplitude) * tick_increment;
    float delta_high_amplitude = (self.high_amplitude_ - current_high_amplitude) * tick_increment;

    float const size_mult = std::exp2(param.size * kSizePowerRange + kMinSizePower);

    float const decay_samples = param.decay_ms * kBaseSampleRate / 1000.0f;
    float const decay_period = size_mult / decay_samples;
    simd::Float256 current_decay1 = self.decays_[0];
    simd::Float256 current_decay2 = self.decays_[1];
    if (param.freeze) {
        self.decays_.fill(simd::BroadcastF256(1.0f));
    }
    else {
        for (size_t j = 0; j < self.kContainerSize; ++j) {
            for (size_t i = 0; i < 8; ++i) {
                self.decays_[j][i] = std::pow(kT60Amplitude, kFeedbackDelays[j * 8 + i] * decay_period);
            }
        }
    }
    auto delta_decay1 = (self.decays_[0] - current_decay1) * tick_increment;
    auto delta_decay2 = (self.decays_[1] - current_decay2) * tick_increment;

    const simd::Int256* allpass_delays = (const simd::Int256*)kAllpassDelays.data();
    auto allpass_offset1 = simd::ToInt(simd::ToFloat(allpass_delays[0]) * self.buffer_scale_ratio_);
    auto allpass_offset2 = simd::ToInt(simd::ToFloat(allpass_delays[1]) * self.buffer_scale_ratio_);

    float const chorus_phase_increment = param.chorus_freq / self.fs_;

    float const network_offset = 2.0f * std::numbers::pi_v<float> / kNetworkSize;
    auto phase_offset = simd::Float128{0.0f, 1.0f, 2.0f, 3.0f} * network_offset;
    auto container_phase = phase_offset + self.chorus_phase_ * 2.0f * std::numbers::pi_v<float>;
    self.chorus_phase_ += static_cast<float>(num_samples) * chorus_phase_increment;
    self.chorus_phase_ -= std::floor(self.chorus_phase_);

    auto chorus_increment_real =
        simd::BroadcastF128(std::cos(chorus_phase_increment * (2.0f * std::numbers::pi_v<float>)));
    auto chorus_increment_imaginary =
        simd::BroadcastF128(std::sin(chorus_phase_increment * (2.0f * std::numbers::pi_v<float>)));
    simd::Float128 current_chorus_real;
    simd::Float128 current_chorus_imaginary;
    for (size_t i = 0; i < 4; ++i) {
        current_chorus_real[i] = std::cos(container_phase[i]);
    }
    for (size_t i = 0; i < 4; ++i) {
        current_chorus_imaginary[i] = std::sin(container_phase[i]);
    }

    const simd::Float256* feedback_delays = (const simd::Float256*)kFeedbackDelays.data();
    auto delay1 = feedback_delays[0] * (self.fs_ / kBaseSampleRate * size_mult);
    auto delay2 = feedback_delays[1] * (self.fs_ / kBaseSampleRate * size_mult);

    auto current_chorus_amount = self.chorus_amount_;
    self.chorus_amount_ = simd::BroadcastF256(param.chorus_amount * kMaxChorusDrift * self.fs_ratio_);
    self.chorus_amount_ = simd::Min(self.chorus_amount_, delay1 - 8 * simd::LaneSize<simd::Float128>);
    self.chorus_amount_ = simd::Min(self.chorus_amount_, delay2 - 8 * simd::LaneSize<simd::Float128>);
    auto delta_chorus_amount = (self.chorus_amount_ - current_chorus_amount) * tick_increment;
    current_chorus_amount = current_chorus_amount * size_mult;

    float current_sample_delay = self.sample_delay_;
    float current_delay_increment = self.sample_delay_increment_;
    float end_target = current_sample_delay + current_delay_increment * static_cast<float>(num_samples);
    float target_delay = std::max(kMinDelay, param.pre_delay * self.fs_ / 1000.0f);
    target_delay = std::lerp(self.sample_delay_, target_delay, kSampleDelayMultiplier);
    float makeup_delay = target_delay - end_target;
    float delta_delay_increment =
        makeup_delay / (0.5f * static_cast<float>(num_samples * num_samples)) * kSampleIncrementMultiplier;

    auto feedback_offset1 = self.feedback_offsets_[0];
    auto feedback_offset2 = self.feedback_offsets_[1];
    float const feedback_smooth = self.feedback_offset_smooth_factor_;

    float input_gain = param.freeze ? 0.0f : 1.0f;

    for (int i = 0; i < num_samples; ++i) {
        // paralle chorus delaylines
        current_chorus_amount += delta_chorus_amount;
        auto const prev_chorus_real = current_chorus_real;
        current_chorus_real =
            prev_chorus_real * chorus_increment_real - current_chorus_imaginary * chorus_increment_imaginary;
        current_chorus_imaginary =
            current_chorus_imaginary * chorus_increment_real + prev_chorus_real * chorus_increment_imaginary;
        auto new_feedback_offset1 =
            delay1 + simd::Combine(current_chorus_real, -current_chorus_real) * current_chorus_amount;
        auto new_feedback_offset2 =
            delay2 + simd::Combine(current_chorus_imaginary, -current_chorus_imaginary) * current_chorus_amount;
        feedback_offset1 += feedback_smooth * (new_feedback_offset1 - feedback_offset1);
        feedback_offset2 += feedback_smooth * (new_feedback_offset2 - feedback_offset2);

        auto feedback_read1 = self.ReadFeedback(0, feedback_offset1);
        auto feedback_read2 = self.ReadFeedback(1, feedback_offset2);

        simd::Float128 input;
        if constexpr (kMono) {
            input = simd::Float128{left[i], left[i], left[i], left[i]};
        }
        else {
            input = simd::Float128{left[i], right[i], left[i], right[i]};
        }
        auto pre_high_coeff_v = simd::BroadcastF128(current_high_pre_coefficient);
        auto pre_low_coeff_v = simd::BroadcastF128(current_low_pre_coefficient);
        auto filtered_input = self.high_pre_filter_.TickLowpass(input * input_gain, pre_high_coeff_v);
        filtered_input = self.low_pre_filter_.TickLowpass(input, pre_low_coeff_v) - filtered_input;
        auto scaled_input = simd::Combine(filtered_input, filtered_input) * 0.5f;

        // paralle polyphase allpass
        auto allpass_read1 = self.ReadAllpass(0, allpass_offset1);
        auto allpass_read2 = self.ReadAllpass(1, allpass_offset2);

        auto allpass_delay_input1 = feedback_read1 - allpass_read1 * kAllpassFeedback;
        auto allpass_delay_input2 = feedback_read2 - allpass_read2 * kAllpassFeedback;

        self.allpass_lookups_[0][static_cast<size_t>(self.allpass_write_pos_)] = scaled_input + allpass_delay_input1;
        self.allpass_lookups_[1][static_cast<size_t>(self.allpass_write_pos_)] = scaled_input + allpass_delay_input2;
        self.allpass_write_pos_ = (self.allpass_write_pos_ + 1) & self.poly_allpass_mask_;

        auto allpass_output1 = allpass_read1 + allpass_delay_input1 * kAllpassFeedback;
        auto allpass_output2 = allpass_read2 + allpass_delay_input2 * kAllpassFeedback;

        // scatter matrix
        simd::Float256 write1;
        simd::Float256 write2;
        _ScatterLane8(allpass_output1, allpass_output2, write1, write2);

        // damp filter
        auto high_coeff_v = simd::BroadcastF256(current_high_coefficient);
        auto low_coeff_v = simd::BroadcastF256(current_low_coefficient);
        auto high_filtered1 = self.high_shelf_filters_[0].TickLowpass(write1, high_coeff_v);
        auto high_filtered2 = self.high_shelf_filters_[1].TickLowpass(write2, high_coeff_v);
        write1 = high_filtered1 + (current_high_amplitude) * (write1 - high_filtered1);
        write2 = high_filtered2 + (current_high_amplitude) * (write2 - high_filtered2);

        auto low_filtered1 = self.low_shelf_filters_[0].TickLowpass(write1, low_coeff_v);
        auto low_filtered2 = self.low_shelf_filters_[1].TickLowpass(write2, low_coeff_v);
        write1 -= low_filtered1 * (current_low_amplitude);
        write2 -= low_filtered2 * (current_low_amplitude);

        // decay block
        current_decay1 += delta_decay1;
        current_decay2 += delta_decay2;
        auto store1 = current_decay1 * write1;
        auto store2 = current_decay2 * write2;
        self.write_index_ = (self.write_index_ + 1) & self.feedback_mask_;
        self.PushFeedback(store1, store2);

        // scatter matrix
        simd::Float256 feed_forward1;
        simd::Float256 feed_forward2;
        _ScatterLane8(store1, store2, feed_forward1, feed_forward2);

        // predelay
        auto total = write1 + write2;
        total += (feed_forward1 * current_decay1 + feed_forward2 * current_decay2) * (0.125f);

        simd::Float128 output{total[0] + total[2] + total[4] + total[6], total[1] + total[3] + total[5] + total[7]};
        self.predelay_.Push(output);
        auto audio_out = current_wet * self.predelay_.GetAfterPush(current_sample_delay) + current_dry * input;
        left[i] = audio_out[0];
        if constexpr (!kMono) {
            right[i] = audio_out[1];
        }

        current_delay_increment += delta_delay_increment;
        current_sample_delay += current_delay_increment;
        current_sample_delay = std::max(current_sample_delay, kMinDelay);
        current_dry += delta_dry;
        current_wet += delta_wet;
        current_high_coefficient += delta_high_coefficient;
        current_high_amplitude += delta_high_amplitude;
        current_low_pre_coefficient += delta_low_pre_coefficient;
        current_high_pre_coefficient += delta_high_pre_coefficient;
        current_low_coefficient += delta_low_coefficient;
        current_low_amplitude += delta_low_amplitude;
    }

    self.sample_delay_increment_ = current_delay_increment;
    self.sample_delay_ = current_sample_delay;
    self.feedback_offsets_[0] = feedback_offset1;
    self.feedback_offsets_[1] = feedback_offset2;
}

static void Process(dsp::ProcessorState& state, float* left, float* right, int num_samples) noexcept {
    if (right == nullptr) {
        ProcessInternal<true>(state, left, right, num_samples);
    }
    else {
        ProcessInternal<false>(state, left, right, num_samples);
    }
}

static void Panic(dsp::ProcessorState& state) noexcept {
    state.lane8.Panic();
}

// ----------------------------------------
// export
// ----------------------------------------

#ifndef DSP_EXPORT_NAME
#error "不应该编译这个文件,在其他cpp包含这个cpp并定义DSP_EXPORT_NAME=`dsp_dispatch.cpp里的变量`"
#endif

ProcessorDsp DSP_EXPORT_NAME{Init, Reset, Panic, Update, Process, DSP_INST_NAME};
} // namespace dsp
