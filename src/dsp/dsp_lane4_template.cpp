#include "dsp_state.hpp"

#include <qwqdsp/convert.hpp>

namespace dsp {

// ----------------------------------------
// dsp processor
// ----------------------------------------

static void Init(dsp::ProcessorState& state, float fs) noexcept {
    auto& self = state.lane4;

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
    auto& self = state.lane4;
    const auto& param = state.param;

    self.wet_ = 0;
    self.dry_ = 0;
    self.chorus_amount_ = simd::BroadcastF128(param.chorus_amount * kMaxChorusDrift);

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
        d = simd::BroadcastF128(0);
    }

    const simd::Float128* feedback_delays = (const simd::Float128*)kFeedbackDelays.data();
    for (size_t i = 0; i < self.kContainerSize; ++i) {
        self.feedback_offsets_[i] = feedback_delays[i];
    }

    for (auto& buffer : self.allpass_lookups_) {
        std::fill(buffer.begin(), buffer.end(), simd::Float128{});
    }
    std::fill(self.feedback_memorie_.begin(), self.feedback_memorie_.end(), 0.0f);
}

static void Update(dsp::ProcessorState& state, const dsp::Param& p) noexcept {
    state.param = p;
}

template <bool kMono>
static void ProcessInternal(dsp::ProcessorState& state, float* left, float* right, int num_samples) noexcept {
    auto& self = state.lane4;
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
    simd::Float128 current_decay1 = self.decays_[0];
    simd::Float128 current_decay2 = self.decays_[1];
    simd::Float128 current_decay3 = self.decays_[2];
    simd::Float128 current_decay4 = self.decays_[3];

    if (param.freeze) {
        self.decays_.fill(simd::BroadcastF128(1.0f));
    }
    else {
        for (size_t j = 0; j < self.kContainerSize; ++j) {
            for (size_t i = 0; i < 4; ++i) {
                self.decays_[j][i] = std::pow(kT60Amplitude, kFeedbackDelays[j * 4 + i] * decay_period);
            }
        }
    }
    auto delta_decay1 = (self.decays_[0] - current_decay1) * tick_increment;
    auto delta_decay2 = (self.decays_[1] - current_decay2) * tick_increment;
    auto delta_decay3 = (self.decays_[2] - current_decay3) * tick_increment;
    auto delta_decay4 = (self.decays_[3] - current_decay4) * tick_increment;

    const simd::Int128* allpass_delays = (const simd::Int128*)kAllpassDelays.data();
    auto allpass_offset1 = simd::ToInt(simd::ToFloat(allpass_delays[0]) * self.buffer_scale_ratio_);
    auto allpass_offset2 = simd::ToInt(simd::ToFloat(allpass_delays[1]) * self.buffer_scale_ratio_);
    auto allpass_offset3 = simd::ToInt(simd::ToFloat(allpass_delays[2]) * self.buffer_scale_ratio_);
    auto allpass_offset4 = simd::ToInt(simd::ToFloat(allpass_delays[3]) * self.buffer_scale_ratio_);

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

    const simd::Float128* feedback_delays = (const simd::Float128*)kFeedbackDelays.data();
    auto delay1 = feedback_delays[0] * (self.fs_ / kBaseSampleRate * size_mult);
    auto delay2 = feedback_delays[1] * (self.fs_ / kBaseSampleRate * size_mult);
    auto delay3 = feedback_delays[2] * (self.fs_ / kBaseSampleRate * size_mult);
    auto delay4 = feedback_delays[3] * (self.fs_ / kBaseSampleRate * size_mult);

    auto current_chorus_amount = self.chorus_amount_;
    self.chorus_amount_ = simd::BroadcastF128(param.chorus_amount * kMaxChorusDrift * self.fs_ratio_);
    self.chorus_amount_ = simd::Min(self.chorus_amount_, delay1 - 8 * simd::LaneSize<simd::Float128>);
    self.chorus_amount_ = simd::Min(self.chorus_amount_, delay2 - 8 * simd::LaneSize<simd::Float128>);
    self.chorus_amount_ = simd::Min(self.chorus_amount_, delay3 - 8 * simd::LaneSize<simd::Float128>);
    self.chorus_amount_ = simd::Min(self.chorus_amount_, delay4 - 8 * simd::LaneSize<simd::Float128>);
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
    auto feedback_offset3 = self.feedback_offsets_[2];
    auto feedback_offset4 = self.feedback_offsets_[3];

    float input_gain = param.freeze ? 0.0f : 1.0f;

    for (int i = 0; i < num_samples; ++i) {
        // paralle chorus delaylines
        current_chorus_amount += delta_chorus_amount;
        current_chorus_real =
            current_chorus_real * chorus_increment_real - current_chorus_imaginary * chorus_increment_imaginary;
        current_chorus_imaginary =
            current_chorus_imaginary * chorus_increment_real + current_chorus_real * chorus_increment_imaginary;
        auto new_feedback_offset1 = delay1 + current_chorus_real * current_chorus_amount;
        auto new_feedback_offset2 = delay2 - current_chorus_real * current_chorus_amount;
        auto new_feedback_offset3 = delay3 + current_chorus_imaginary * current_chorus_amount;
        auto new_feedback_offset4 = delay4 - current_chorus_imaginary * current_chorus_amount;
        feedback_offset1 += (self.feedback_offset_smooth_factor_) * (new_feedback_offset1 - feedback_offset1);
        feedback_offset2 += (self.feedback_offset_smooth_factor_) * (new_feedback_offset2 - feedback_offset2);
        feedback_offset3 += (self.feedback_offset_smooth_factor_) * (new_feedback_offset3 - feedback_offset3);
        feedback_offset4 += (self.feedback_offset_smooth_factor_) * (new_feedback_offset4 - feedback_offset4);

        auto feedback_read1 = self.ReadFeedback(0, feedback_offset1);
        auto feedback_read2 = self.ReadFeedback(1, feedback_offset2);
        auto feedback_read3 = self.ReadFeedback(2, feedback_offset3);
        auto feedback_read4 = self.ReadFeedback(3, feedback_offset4);

        simd::Float128 input;
        if constexpr (kMono) {
            input = simd::Float128{left[i], left[i], left[i], left[i]};
        }
        else {
            input = simd::Float128{left[i], right[i], left[i], right[i]};
        }
        auto filtered_input = self.high_pre_filter_.TickLowpass(input * input_gain, simd::BroadcastF128(current_high_pre_coefficient));
        filtered_input = self.low_pre_filter_.TickLowpass(input, simd::BroadcastF128(current_low_pre_coefficient)) - filtered_input;
        auto scaled_input = filtered_input * 0.5f;

        // paralle polyphase allpass
        auto allpass_read1 = self.ReadAllpass(0, allpass_offset1);
        auto allpass_read2 = self.ReadAllpass(1, allpass_offset2);
        auto allpass_read3 = self.ReadAllpass(2, allpass_offset3);
        auto allpass_read4 = self.ReadAllpass(3, allpass_offset4);

        auto allpass_delay_input1 = feedback_read1 - allpass_read1 * kAllpassFeedback;
        auto allpass_delay_input2 = feedback_read2 - allpass_read2 * kAllpassFeedback;
        auto allpass_delay_input3 = feedback_read3 - allpass_read3 * kAllpassFeedback;
        auto allpass_delay_input4 = feedback_read4 - allpass_read4 * kAllpassFeedback;

        self.allpass_lookups_[0][static_cast<size_t>(self.allpass_write_pos_)] = scaled_input + allpass_delay_input1;
        self.allpass_lookups_[1][static_cast<size_t>(self.allpass_write_pos_)] = scaled_input + allpass_delay_input2;
        self.allpass_lookups_[2][static_cast<size_t>(self.allpass_write_pos_)] = scaled_input + allpass_delay_input3;
        self.allpass_lookups_[3][static_cast<size_t>(self.allpass_write_pos_)] = scaled_input + allpass_delay_input4;
        self.allpass_write_pos_ = (self.allpass_write_pos_ + 1) & self.poly_allpass_mask_;

        auto allpass_output1 = allpass_read1 + allpass_delay_input1 * kAllpassFeedback;
        auto allpass_output2 = allpass_read2 + allpass_delay_input2 * kAllpassFeedback;
        auto allpass_output3 = allpass_read3 + allpass_delay_input3 * kAllpassFeedback;
        auto allpass_output4 = allpass_read4 + allpass_delay_input4 * kAllpassFeedback;

        // scatter matrix
        // write1 = 0.25 * Sum(All) + Ao0 - 0.5 * Sum(Ao0) - 0.5 * (Ao0 + Ao1 + Ao2 + Ao3)
        // write2 = 0.25 * Sum(All) + Ao1 - 0.5 * Sum(Ao1) - 0.5 * (Ao0 + Ao1 + Ao2 + Ao3)
        // write3 = 0.25 * Sum(All) + Ao2 - 0.5 * Sum(Ao2) - 0.5 * (Ao0 + Ao1 + Ao2 + Ao3)
        // write4 = 0.25 * Sum(All) + Ao3 - 0.5 * Sum(Ao3) - 0.5 * (Ao0 + Ao1 + Ao2 + Ao3)
        auto total_rows = allpass_output1 + allpass_output2 + allpass_output3 + allpass_output4;
        auto other_feedback = total_rows * (-0.5f) + (simd::ReduceAdd(total_rows) * 0.25f);

        auto write1 = other_feedback + allpass_output1;
        auto write2 = other_feedback + allpass_output2;
        auto write3 = other_feedback + allpass_output3;
        auto write4 = other_feedback + allpass_output4;

        alignas(16) auto [t1, t2, t3, t4] = simd::Transpose(allpass_output1, allpass_output2, allpass_output3, allpass_output4);
        auto adjacent_feedback = (t1 + t2 + t3 + t4) * (-0.5f);
        write1 += adjacent_feedback[0];
        write2 += adjacent_feedback[1];
        write3 += adjacent_feedback[2];
        write4 += adjacent_feedback[3];

        // damp filter
        auto high_filtered1 = self.high_shelf_filters_[0].TickLowpass(write1, simd::BroadcastF128(current_high_coefficient));
        auto high_filtered2 = self.high_shelf_filters_[1].TickLowpass(write2, simd::BroadcastF128(current_high_coefficient));
        auto high_filtered3 = self.high_shelf_filters_[2].TickLowpass(write3, simd::BroadcastF128(current_high_coefficient));
        auto high_filtered4 = self.high_shelf_filters_[3].TickLowpass(write4, simd::BroadcastF128(current_high_coefficient));
        write1 = high_filtered1 + (current_high_amplitude) * (write1 - high_filtered1);
        write2 = high_filtered2 + (current_high_amplitude) * (write2 - high_filtered2);
        write3 = high_filtered3 + (current_high_amplitude) * (write3 - high_filtered3);
        write4 = high_filtered4 + (current_high_amplitude) * (write4 - high_filtered4);

        auto low_filtered1 = self.low_shelf_filters_[0].TickLowpass(write1, simd::BroadcastF128(current_low_coefficient));
        auto low_filtered2 = self.low_shelf_filters_[1].TickLowpass(write2, simd::BroadcastF128(current_low_coefficient));
        auto low_filtered3 = self.low_shelf_filters_[2].TickLowpass(write3, simd::BroadcastF128(current_low_coefficient));
        auto low_filtered4 = self.low_shelf_filters_[3].TickLowpass(write4, simd::BroadcastF128(current_low_coefficient));
        write1 -= low_filtered1 * (current_low_amplitude);
        write2 -= low_filtered2 * (current_low_amplitude);
        write3 -= low_filtered3 * (current_low_amplitude);
        write4 -= low_filtered4 * (current_low_amplitude);

        // decay block
        current_decay1 += delta_decay1;
        current_decay2 += delta_decay2;
        current_decay3 += delta_decay3;
        current_decay4 += delta_decay4;
        auto store1 = current_decay1 * write1;
        auto store2 = current_decay2 * write2;
        auto store3 = current_decay3 * write3;
        auto store4 = current_decay4 * write4;
        self.write_index_ = (self.write_index_ + 1) & self.feedback_mask_;
        self.feedback_ptrs_[0][self.write_index_] = store1[0];
        self.feedback_ptrs_[1][self.write_index_] = store1[1];
        self.feedback_ptrs_[2][self.write_index_] = store1[2];
        self.feedback_ptrs_[3][self.write_index_] = store1[3];
        self.feedback_ptrs_[4][self.write_index_] = store2[0];
        self.feedback_ptrs_[5][self.write_index_] = store2[1];
        self.feedback_ptrs_[6][self.write_index_] = store2[2];
        self.feedback_ptrs_[7][self.write_index_] = store2[3];
        self.feedback_ptrs_[8][self.write_index_] = store3[0];
        self.feedback_ptrs_[9][self.write_index_] = store3[1];
        self.feedback_ptrs_[10][self.write_index_] = store3[2];
        self.feedback_ptrs_[11][self.write_index_] = store3[3];
        self.feedback_ptrs_[12][self.write_index_] = store4[0];
        self.feedback_ptrs_[13][self.write_index_] = store4[1];
        self.feedback_ptrs_[14][self.write_index_] = store4[2];
        self.feedback_ptrs_[15][self.write_index_] = store4[3];

        // scatter matrix
        auto total_allpass = store1 + store2 + store3 + store4;
        auto other_feedback_allpass = total_allpass * (-0.5f) + (simd::ReduceAdd(total_allpass) * 0.25f);

        auto feed_forward1 = other_feedback_allpass + store1;
        auto feed_forward2 = other_feedback_allpass + store2;
        auto feed_forward3 = other_feedback_allpass + store3;
        auto feed_forward4 = other_feedback_allpass + store4;

        alignas(16) auto [s1, s2, s3, s4] = simd::Transpose(store1, store2, store3, store4);
        auto adjacent_feedback_allpass = (s1 + s2 + s3 + s4) * (-0.5f);

        feed_forward1 += (adjacent_feedback_allpass[0]);
        feed_forward2 += (adjacent_feedback_allpass[1]);
        feed_forward3 += (adjacent_feedback_allpass[2]);
        feed_forward4 += (adjacent_feedback_allpass[3]);

        // predelay
        auto total = write1 + write2 + write3 + write4;
        total += (feed_forward1 * current_decay1 + feed_forward2 * current_decay2 + feed_forward3 * current_decay3
                  + feed_forward4 * current_decay4)
               * (0.125f);

        simd::Float128 output{total[0] + total[2], total[1] + total[3]};
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
    self.feedback_offsets_[2] = feedback_offset3;
    self.feedback_offsets_[3] = feedback_offset4;
}

static void Process(dsp::ProcessorState& state, float* left, float* right, int num_samples) noexcept {
    if (right == nullptr) {
        ProcessInternal<true>(state, left, right, num_samples);
    }
    else {
        ProcessInternal<false>(state, left, right, num_samples);
    }
}

// ----------------------------------------
// export
// ----------------------------------------

#ifndef DSP_EXPORT_NAME
#error "不应该编译这个文件,在其他cpp包含这个cpp并定义DSP_EXPORT_NAME=`dsp_dispatch.cpp里的变量`"
#endif

ProcessorDsp DSP_EXPORT_NAME{Init, Reset, Update, Process, DSP_INST_NAME};
} // namespace dsp
