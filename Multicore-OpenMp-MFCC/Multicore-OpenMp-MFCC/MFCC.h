#pragma once
#include "FFT.h"

class MFCC
{
private:
    int          n_mfcc;
    double       fMin;
    int          n_fft;
    int          hop_length;
    int	        n_mels;

    double       sampleRate;
    double       fMax;

    FFT fft = FFT();
    
public:
    MFCC();
    void setSampleRate(double sampleRateVal) { sampleRate = sampleRateVal; }

    void setN_mfcc(int n_mfccVal) { n_mfcc = n_mfccVal; }

    vector<float> process(vector<double>& doubleInputBuffer);

    //MFCC into 1d
    vector<float> finalshape(vector<vector<double>>& mfccSpecTro);

    //DCT to mfcc, librosa
    vector<vector<double>> dctMfcc(vector<double>& y);

    //mel spectrogram, librosa
    vector<vector<double>> melSpectrogram(vector<double>& y);

    //stft, librosa
    vector<vector<double>> stftMagSpec(vector<double>& y);
    vector<double> magSpectrogram(vector<double>& frame);

    //get hann window, librosa
    vector<double> getWindow();

    //frame, librosa
    vector<vector<double>> yFrame(vector<double>& ypad);

    //power to db, librosa
    vector<vector<double>> powerToDb(vector<vector<double>>& melS);

    //dct, librosa
    vector<vector<double>> dctFilter(int n_filters, int n_input);

    //mel, librosa
    vector<vector<double>> melFilter();

    //fft frequencies, librosa
    vector<double> fftFreq();

    //mel frequencies, librosa
    vector<double> melFreq(int numMels);

    //mel to hz, htk, librosa
    vector<double> melToFreqS(vector<double>& mels);

    // hz to mel, htk, librosa
    vector<double> freqToMelS(vector<double>& freqs);

    //mel to hz, Slaney, librosa
    vector<double> melToFreq(vector<double>& mels);

    // hz to mel, Slaney, librosa
    vector<double> freqToMel(vector<double>& freqs);

    // log10
    double log10(double value) { return log(value) / log(10); }
};

