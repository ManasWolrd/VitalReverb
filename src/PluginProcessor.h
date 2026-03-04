#pragma once
#include "pluginshared/juce_param_listener.hpp"
#include "pluginshared/preset_manager.hpp"
#include "pluginshared/wrap_parameters.hpp"

#include "dsp/dsp_state.hpp"

class EmptyAudioProcessor final : public juce::AudioProcessor {
public:
    static constexpr auto kParameterValueTreeIdentify = "PARAMETERS";
    //==============================================================================
    EmptyAudioProcessor();
    ~EmptyAudioProcessor() override;

    //==============================================================================
    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;

    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;

    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;
    using AudioProcessor::processBlock;

    //==============================================================================
    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    //==============================================================================
    const juce::String getName() const override;

    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;

    //==============================================================================
    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram(int index) override;
    const juce::String getProgramName(int index) override;
    void changeProgramName(int index, const juce::String& newName) override;

    //==============================================================================
    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;

    JuceParamListener param_listener_;
    std::unique_ptr<juce::AudioProcessorValueTreeState> value_tree_;
    std::unique_ptr<pluginshared::PresetManager> preset_manager_;

    juce::AudioParameterFloat* param_chorus_amount_;
    juce::AudioParameterFloat* param_chorus_freq_;
    juce::AudioParameterFloat* param_wet_;
    juce::AudioParameterFloat* param_pre_lowpass_;
    juce::AudioParameterFloat* param_pre_highpass_;
    juce::AudioParameterFloat* param_low_damp_pitch_;
    juce::AudioParameterFloat* param_high_damp_pitch_;
    juce::AudioParameterFloat* param_low_damp_db_;
    juce::AudioParameterFloat* param_high_damp_db_;
    juce::AudioParameterFloat* param_size_;
    juce::AudioParameterFloat* param_decay_ms_;
    juce::AudioParameterFloat* param_predelay_;

    dsp::ProcessorDsp dsp_processor_;
    dsp::ProcessorState dsp_state_;
    dsp::Param dsp_param_;
private:
    //==============================================================================
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(EmptyAudioProcessor)
};
