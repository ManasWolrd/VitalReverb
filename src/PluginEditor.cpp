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
static_assert(std::is_base_of_v<juce::Component, PluginUi>, "PluginUi must inherit from juce::Component");

EmptyAudioProcessorEditor::EmptyAudioProcessorEditor(EmptyAudioProcessor& p)
    : AudioProcessorEditor(&p)
    , ui_(p) {
    auto ui_bound = ui_.getLocalBounds();
    if (ui_bound.isEmpty()) {
        jassert("you must set an editor size" && false);
    }
    ui_width_ = ui_bound.getWidth();
    ui_height_ = ui_bound.getHeight();

    auto* props = plugin_config_->config.get();
    if (props != nullptr) {
        scale_ = static_cast<float>(props->getDoubleValue("scale", scale_));
        setSize(static_cast<int>(static_cast<float>(ui_width_) * scale_), static_cast<int>(static_cast<float>(ui_height_) * scale_));
    }
    else {
        setSize(ui_width_, ui_height_);
    }
    setResizable(true, true);
    getConstrainer()->setFixedAspectRatio(static_cast<float>(ui_width_) / static_cast<float>(ui_height_));
    setResizeLimits(ui_width_, ui_height_, 9999, 9999);

    ui_.on_want_new_size = [this](int width, int height) {
        ui_width_ = width;
        ui_height_ = height;
        ui_.setSize(width, height);
        getConstrainer()->setFixedAspectRatio(static_cast<float>(ui_width_) / static_cast<float>(ui_height_));
        setSize(static_cast<int>(static_cast<float>(width) * scale_), static_cast<int>(static_cast<float>(height) * scale_));
    };
    addAndMakeVisible(ui_);
}

EmptyAudioProcessorEditor::~EmptyAudioProcessorEditor() {
}

//==============================================================================
void EmptyAudioProcessorEditor::paint(juce::Graphics& g) {
    g.fillAll(ui::green_bg);
}

void EmptyAudioProcessorEditor::resized() {
    scale_ = static_cast<float>(getWidth()) / static_cast<float>(ui_width_);
    ui_.setTransform(juce::AffineTransform::scale(scale_, scale_));

    if (auto* props = plugin_config_->config.get()) {
        if (getWidth() > 0 && getHeight() > 0) {
            props->setValue("scale", scale_);
        }
    }
}
