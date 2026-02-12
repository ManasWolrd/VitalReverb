#pragma once

#include "ui/plugin_ui.hpp"

//==============================================================================
class EmptyAudioProcessorEditor final : public juce::AudioProcessorEditor {
public:
    explicit EmptyAudioProcessorEditor (EmptyAudioProcessor&);
    ~EmptyAudioProcessorEditor() override;

    //==============================================================================
    void paint (juce::Graphics&) override;
    void resized() override;

private:
    PluginUi ui_;

    struct PluginConfig;
    juce::SharedResourcePointer<PluginConfig> plugin_config_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (EmptyAudioProcessorEditor)
};
