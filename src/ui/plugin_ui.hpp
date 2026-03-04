#pragma once

#include "pluginshared/component.hpp"
#include "pluginshared/preset_panel.hpp"

class EmptyAudioProcessor;

class PluginUi : public juce::Component {
public:
    static constexpr int kWidth = 480;
    static constexpr int kHeight = 320;

    explicit PluginUi(EmptyAudioProcessor& p);

    void resized() override;
    void paint(juce::Graphics& g) override;
    
    std::function<void(int,int)> on_want_new_size;
private:
    void TrySetSize(int width, int height) {
        if (on_want_new_size) {
            on_want_new_size(width, height); 
        }
    }
    pluginshared::PresetPanel preset_;

    ui::Dial chorus_amount_{"CHOR AMT"};
    ui::Dial chorus_freq_{"CHOR FREQ"};
    ui::Dial mix_{"MIX"};
    ui::Dial pre_lowpass_{"LOW CUT"};
    ui::Dial pre_highpass_{"HIGH CUT"};
    ui::Dial low_damp_{"LOW DAMP"};
    ui::Dial high_damp_{"HIGH DAMP"};
    ui::Dial low_gain_{"LOW GAIN"};
    ui::Dial high_gain_{"HIGH GAIN"};
    ui::Dial size_{"SIZE"};
    ui::Dial decay_{"TIME"};
    ui::Dial predelay_{"DELAY"};
};
