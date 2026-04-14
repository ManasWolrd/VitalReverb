#include "PluginProcessor.h"
#include "PluginEditor.h"
#include "global.hpp"

//==============================================================================
EmptyAudioProcessor::EmptyAudioProcessor()
    : AudioProcessor(BusesProperties()
#if !JucePlugin_IsMidiEffect
#if !JucePlugin_IsSynth
                         .withInput("Input", juce::AudioChannelSet::stereo(), true)
#endif
                         .withOutput("Output", juce::AudioChannelSet::stereo(), true)
#endif
      ) {
    dsp_processor_ = dsp::GetProcessorDsp();

    juce::AudioProcessorValueTreeState::ParameterLayout layout;

    {
        auto p = std::make_unique<juce::AudioParameterFloat>(
            juce::ParameterID{"chorus amount", 1},
            "chorus amount",
            0.0f, 1.0f, 0.05f
        );
        param_chorus_amount_ = p.get();
        layout.add(std::move(p));
    }
    {
        auto p = std::make_unique<juce::AudioParameterFloat>(
            juce::ParameterID{"chorus freq", 1},
            "chorus freq",
            juce::NormalisableRange<float>{0.003f, 8.0f, 0.001f, 0.4f},
            0.25f
        );
        param_chorus_freq_ = p.get();
        layout.add(std::move(p));
    }
    {
        auto p = std::make_unique<juce::AudioParameterFloat>(
            juce::ParameterID{"mix", 1},
            "mix",
            0.0f, 1.0f, 0.25f
        );
        param_wet_ = p.get();
        layout.add(std::move(p));
    }
    {
        auto p = std::make_unique<juce::AudioParameterFloat>(
            juce::ParameterID{"pre lowpass", 1},
            "pre lowpass",
            0.0f, 130.0f, 0.0f
        );
        param_pre_lowpass_ = p.get();
        layout.add(std::move(p));
    }
    {
        auto p = std::make_unique<juce::AudioParameterFloat>(
            juce::ParameterID{"pre highpass", 1},
            "pre highpass",
            0.0f, 130.0f, 110.0f
        );
        param_pre_highpass_ = p.get();
        layout.add(std::move(p));
    }
    {
        auto p = std::make_unique<juce::AudioParameterFloat>(
            juce::ParameterID{"low damp", 1},
            "low damp",
            0.0f, 130.0f, 0.0f
        );
        param_low_damp_pitch_ = p.get();
        layout.add(std::move(p));
    }
    {
        auto p = std::make_unique<juce::AudioParameterFloat>(
            juce::ParameterID{"high damp", 1},
            "high damp",
            0.0f, 130.0f, 90.0f
        );
        param_high_damp_pitch_ = p.get();
        layout.add(std::move(p));
    }
    {
        auto p = std::make_unique<juce::AudioParameterFloat>(
            juce::ParameterID{"low gain", 1},
            "low gain",
            -6.0f, 0.0f, 0.0f
        );
        param_low_damp_db_ = p.get();
        layout.add(std::move(p));
    }
    {
        auto p = std::make_unique<juce::AudioParameterFloat>(
            juce::ParameterID{"high gain", 1},
            "high gain",
            -6.0f, 0.0f, -1.0f
        );
        param_high_damp_db_ = p.get();
        layout.add(std::move(p));
    }
    {
        auto p = std::make_unique<juce::AudioParameterFloat>(
            juce::ParameterID{"size", 1},
            "size",
            0.0f, 1.0f, 0.5f
        );
        param_size_ = p.get();
        layout.add(std::move(p));
    }
    {
        auto p = std::make_unique<juce::AudioParameterFloat>(
            juce::ParameterID{"decay", 1},
            "decay",
            juce::NormalisableRange<float>{15.0f, 64000.0f, 1.0f, 0.4f},
            1000.0f
        );
        param_decay_ms_ = p.get();
        layout.add(std::move(p));
    }
    {
        auto p = std::make_unique<juce::AudioParameterFloat>(
            juce::ParameterID{"predelay", 1},
            "predelay",
            juce::NormalisableRange<float>{0.0f, 300.0f, 1.0f, 0.4f},
            0.0f
        );
        param_predelay_ = p.get();
        layout.add(std::move(p));
    }
    {
        auto p = std::make_unique<juce::AudioParameterBool>(
            juce::ParameterID{"freeze", 1},
            "freeze",
            false
        );
        param_freeze_ = p.get();
        layout.add(std::move(p));
    }

    value_tree_ = std::make_unique<juce::AudioProcessorValueTreeState>(*this, nullptr, kParameterValueTreeIdentify,
                                                                       std::move(layout));
    preset_manager_ = std::make_unique<pluginshared::PresetManager>(*value_tree_, *this, pluginshared::UpdateData::GithubInfo{
        global::kPluginRepoOwnerName, global::kPluginRepoName
    });
}

EmptyAudioProcessor::~EmptyAudioProcessor() {
    param_listener_.Clear();
    preset_manager_ = nullptr;
    value_tree_ = nullptr;
}

//==============================================================================
const juce::String EmptyAudioProcessor::getName() const {
    return JucePlugin_Name;
}

bool EmptyAudioProcessor::acceptsMidi() const {
#if JucePlugin_WantsMidiInput
    return true;
#else
    return false;
#endif
}

bool EmptyAudioProcessor::producesMidi() const {
#if JucePlugin_ProducesMidiOutput
    return true;
#else
    return false;
#endif
}

bool EmptyAudioProcessor::isMidiEffect() const {
#if JucePlugin_IsMidiEffect
    return true;
#else
    return false;
#endif
}

double EmptyAudioProcessor::getTailLengthSeconds() const {
    return 0.0;
}

int EmptyAudioProcessor::getNumPrograms() {
    return 1; // NB: some hosts don't cope very well if you tell them there are 0 programs,
              // so this should be at least 1, even if you're not really implementing programs.
}

int EmptyAudioProcessor::getCurrentProgram() {
    return 0;
}

void EmptyAudioProcessor::setCurrentProgram(int index) {
    juce::ignoreUnused(index);
}

const juce::String EmptyAudioProcessor::getProgramName(int index) {
    juce::ignoreUnused(index);
    return {};
}

void EmptyAudioProcessor::changeProgramName(int index, const juce::String& newName) {
    juce::ignoreUnused(index, newName);
}

//==============================================================================
void EmptyAudioProcessor::prepareToPlay(double sampleRate, int samplesPerBlock) {
    if (!dsp_processor_.IsValid()) return;

    float fs = static_cast<float>(sampleRate);
    dsp_processor_.init(dsp_state_, fs);
    dsp_processor_.reset(dsp_state_);
    param_listener_.MarkAll();
}

void EmptyAudioProcessor::releaseResources() {
    // When playback stops, you can use this as an opportunity to free up any
    // spare memory, etc.
}

void EmptyAudioProcessor::reset() {
    if (dsp_processor_.IsValid()) {
        dsp_processor_.panic(dsp_state_);
    }
}

bool EmptyAudioProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const {
#if JucePlugin_IsMidiEffect
    juce::ignoreUnused(layouts);
    return true;
#else
    // This is the place where you check if the layout is supported.
    // In this template code we only support mono or stereo.
    // Some plugin hosts, such as certain GarageBand versions, will only
    // load plugins that support stereo bus layouts.
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::mono()
        && layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
        return false;

    // This checks if the input layout matches the output layout
#if !JucePlugin_IsSynth
    if (layouts.getMainOutputChannelSet() != layouts.getMainInputChannelSet()) return false;
#endif

    return true;
#endif
}

void EmptyAudioProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages) {
    if (!dsp_processor_.IsValid()) return;

    juce::ScopedNoDenormals noDenormals;
    param_listener_.HandleDirty();

    int const num_samples = buffer.getNumSamples();
    float* left_ptr = buffer.getWritePointer(0);
    float* right_ptr = buffer.getNumChannels() == 2 ? buffer.getWritePointer(1) : nullptr;

    dsp_param_.chorus_amount = param_chorus_amount_->get();
    dsp_param_.chorus_freq = param_chorus_freq_->get();
    dsp_param_.wet = param_wet_->get();
    dsp_param_.pre_lowpass = param_pre_lowpass_->get();
    dsp_param_.pre_highpass = param_pre_highpass_->get();
    dsp_param_.low_damp_pitch = param_low_damp_pitch_->get();
    dsp_param_.high_damp_pitch = param_high_damp_pitch_->get();
    dsp_param_.low_damp_db = param_low_damp_db_->get();
    dsp_param_.high_damp_db = param_high_damp_db_->get();
    dsp_param_.size = param_size_->get();
    dsp_param_.decay_ms = param_decay_ms_->get();
    dsp_param_.pre_delay = param_predelay_->get();
    dsp_param_.freeze = param_freeze_->get();
    dsp_processor_.update(dsp_state_, dsp_param_);

    dsp_processor_.process(dsp_state_, left_ptr, right_ptr, num_samples);
}

//==============================================================================
bool EmptyAudioProcessor::hasEditor() const {
    return true; // (change this to false if you choose to not supply an editor)
}

juce::AudioProcessorEditor* EmptyAudioProcessor::createEditor() {
    return new EmptyAudioProcessorEditor(*this);
    // return new juce::GenericAudioProcessorEditor(*this);
}

//==============================================================================
void EmptyAudioProcessor::getStateInformation(juce::MemoryBlock& destData) {
    suspendProcessing(true);

    juce::ValueTree plugin_state{"PLUGIN_STATE"};
    plugin_state.appendChild(value_tree_->copyState(), nullptr);

    if (auto xml = plugin_state.createXml(); xml != nullptr) {
        copyXmlToBinary(*xml, destData);
    }

    suspendProcessing(false);
}

void EmptyAudioProcessor::setStateInformation(const void* data, int sizeInBytes) {
    suspendProcessing(true);

    auto xml = *getXmlFromBinary(data, sizeInBytes);
    auto plugin_state = juce::ValueTree::fromXml(xml);
    if (plugin_state.isValid()) {
        auto parameter = plugin_state.getChildWithName(kParameterValueTreeIdentify);
        if (parameter.isValid()) {
            value_tree_->replaceState(parameter);
        }
    }

    suspendProcessing(false);
}

//==============================================================================
// This creates new instances of the plugin..
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() {
    return new EmptyAudioProcessor();
}
