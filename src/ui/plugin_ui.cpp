#include "plugin_ui.hpp"

#include "../PluginProcessor.h"

PluginUi::PluginUi(EmptyAudioProcessor& p)
    : preset_(*p.preset_manager_) {
    addAndMakeVisible(preset_);
}

void PluginUi::resized() {
    auto b = getLocalBounds();
    preset_.setBounds(b.removeFromTop(30));
}

void PluginUi::paint(juce::Graphics& g) {
    g.drawText("Plugin UI", getLocalBounds(), juce::Justification::centred);
}
