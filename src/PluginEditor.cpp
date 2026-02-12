#include "PluginEditor.h"

#include "PluginProcessor.h"

struct EmptyAudioProcessorEditor::PluginConfig {
    PluginConfig() {
        juce::PropertiesFile::Options options;
        options.applicationName = JucePlugin_Name;
        options.filenameSuffix = ".settings";
        options.folderName = JucePlugin_Manufacturer;
        options.storageFormat = juce::PropertiesFile::storeAsXML;

        config = std::make_unique<juce::PropertiesFile>(options);
    }

    std::unique_ptr<juce::PropertiesFile> config;
};

// -------------------- plugin editor --------------------

EmptyAudioProcessorEditor::EmptyAudioProcessorEditor(EmptyAudioProcessor& p)
    : AudioProcessorEditor(&p)
    , ui_(p) {
    auto* props = plugin_config_->config.get();

    addAndMakeVisible(ui_);
    if (props != nullptr) {
        float scale_x = static_cast<float>(props->getDoubleValue("scale_x", 1.0));
        float scale_y = static_cast<float>(props->getDoubleValue("scale_y", 1.0));
        setSize(static_cast<int>(ui_.kWidth * scale_x), static_cast<int>(ui_.kHeight * scale_y));
    }
    else {
        setSize(ui_.kWidth, ui_.kHeight);
    }
    setResizable(true, true);
    getConstrainer()->setFixedAspectRatio(static_cast<float>(ui_.kWidth) / ui_.kHeight);
    setResizeLimits(ui_.kWidth, ui_.kHeight, 9999, 9999);
}

EmptyAudioProcessorEditor::~EmptyAudioProcessorEditor() {
}

//==============================================================================
void EmptyAudioProcessorEditor::paint(juce::Graphics& g) {
    g.fillAll(ui::green_bg);
}

void EmptyAudioProcessorEditor::resized() {
    float scaleX = static_cast<float>(getWidth()) / ui_.kWidth;
    float scaleY = static_cast<float>(getHeight()) / ui_.kHeight;
    ui_.setTransform(juce::AffineTransform::scale(scaleX, scaleY));
    ui_.setBounds(0, 0, ui_.kWidth, ui_.kHeight);

    if (auto* props = plugin_config_->config.get()) {
        if (getWidth() > 0 && getHeight() > 0) {
            props->setValue("scale_x", scaleX);
            props->setValue("scale_y", scaleY);
        }
    }
}
