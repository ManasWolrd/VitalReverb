#include "plugin_ui.hpp"

#include "../PluginProcessor.h"

PluginUi::PluginUi(EmptyAudioProcessor& p)
    : preset_(*p.preset_manager_) {
    addAndMakeVisible(preset_);

    chorus_amount_.BindParam(p.param_chorus_amount_);
    addAndMakeVisible(chorus_amount_);
    chorus_freq_.BindParam(p.param_chorus_freq_);
    addAndMakeVisible(chorus_freq_);
    mix_.BindParam(p.param_wet_);
    addAndMakeVisible(mix_);
    pre_lowpass_.BindParam(p.param_pre_lowpass_);
    addAndMakeVisible(pre_lowpass_);
    pre_highpass_.BindParam(p.param_pre_highpass_);
    addAndMakeVisible(pre_highpass_);
    low_damp_.BindParam(p.param_low_damp_pitch_);
    addAndMakeVisible(low_damp_);
    high_damp_.BindParam(p.param_high_damp_pitch_);
    addAndMakeVisible(high_damp_);
    low_gain_.BindParam(p.param_low_damp_db_);
    addAndMakeVisible(low_gain_);
    high_gain_.BindParam(p.param_high_damp_db_);
    addAndMakeVisible(high_gain_);
    size_.BindParam(p.param_size_);
    addAndMakeVisible(size_);
    decay_.BindParam(p.param_decay_ms_);
    addAndMakeVisible(decay_);
    predelay_.BindParam(p.param_predelay_);
    addAndMakeVisible(predelay_);
    freeze_.BindParam(p.param_freeze_);
    addAndMakeVisible(freeze_);

    setSize(500, 230);
}

void PluginUi::resized() {
    auto b = getLocalBounds();
    preset_.setBounds(b.removeFromTop(std::max(30, b.proportionOfHeight(0.1f))));

    auto top = b.removeFromTop(30);
    freeze_.setBounds(top.removeFromLeft(80).reduced(2));

    int w = b.getWidth() / 6;
    top = b.removeFromTop(b.getHeight() / 2);
    pre_lowpass_.setBounds(top.removeFromLeft(w));
    low_damp_.setBounds(top.removeFromLeft(w));
    high_damp_.setBounds(top.removeFromLeft(w));
    chorus_amount_.setBounds(top.removeFromLeft(w));
    predelay_.setBounds(top.removeFromLeft(w));
    mix_.setBounds(top.removeFromLeft(w));
    auto bottom = b;
    pre_highpass_.setBounds(bottom.removeFromLeft(w));
    low_gain_.setBounds(bottom.removeFromLeft(w));
    high_gain_.setBounds(bottom.removeFromLeft(w));
    chorus_freq_.setBounds(bottom.removeFromLeft(w));
    size_.setBounds(bottom.removeFromLeft(w));
    decay_.setBounds(bottom.removeFromLeft(w));
}

void PluginUi::paint(juce::Graphics& g) {
}
