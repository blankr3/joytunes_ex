import os
import glob
import logging
import numpy as np
import scipy.signal
from tqdm import tqdm
import scipy.io.wavfile
import scipy.interpolate
import matplotlib.pyplot as plt

logger = logging.getLogger()

# Inputs
ref_path = r".\data\input\ref_samples.wav"
segment_paths = glob.glob(os.path.join(".\data\input\segments", "validation*.wav"))

# Load data
dataset = dict()
dataset['ref'] = dict()
dataset['ref']['sample_rate'], dataset['ref']['amps'] = scipy.io.wavfile.read(ref_path)  # load ref data
logger.info(f"ref sample rate: {dataset['ref']['sample_rate'] / 1000} KHz")

for p in tqdm(segment_paths):  # load segments data
    name = os.path.basename(p).split('.wav')[0]
    dataset[name] = dict()
    dataset[name]['sample_rate'], dataset[name]['amps'] = scipy.io.wavfile.read(p)


class AudioSegmentFinder():
    def __init__(self, ref_signal, seg_signal, ref_sample_rate, seg_sample_rate):
        self.ref_signal = ref_signal
        self.seg_signal = seg_signal
        self.ref_sample_rate = ref_sample_rate
        self.seg_sample_rate = seg_sample_rate
        self.sample_rate_ratio = ref_sample_rate // seg_sample_rate

        # remvoe DC offsets
        ref_signal -= np.mean(ref_signal)
        seg_signal -= np.mean(seg_signal)

        # downsample ref signal if necessary
        self.ref_signal = self.ref_signal[
                          ::int(self.sample_rate_ratio)]  # if sample rate is identical, this line will have no effect

    @staticmethod
    def find_max_correlation_time(ref_signal, seg_signal):
        ref_signal = ref_signal.copy() / np.max(ref_signal)
        seg_signal = seg_signal.copy() / np.max(seg_signal)
        res = scipy.signal.convolve(ref_signal, seg_signal[::-1],
                                    mode='valid')  # convolve ref and segment arrays (flip the mask array)
        max_index = np.argmax(np.abs(res))  # index of max correlation point
        max_correlation_sample = max_index
        return max_correlation_sample, res

    @staticmethod
    def slice_estimated_segment(ref_signal, seg_signal, _x):
        shifted_x = list(range(_x, len(seg_signal) + _x))
        ref_segment_estimate = ref_signal[shifted_x].copy().astype(float)
        return ref_segment_estimate, shifted_x

    def apply_transfer_function(self):
        ref_fft = np.fft.fft(self.ref_segment_estimate)
        ref_freq = np.fft.fftfreq(len(self.ref_segment_estimate), len(self.ref_segment_estimate) / self.ref_sample_rate)
        seg_fft = np.fft.fft(self.seg_signal)
        seg_freq = np.fft.fftfreq(len(self.seg_signal), len(self.seg_signal) / self.seg_sample_rate)
        distortion_signal = np.fft.ifft(seg_fft / ref_fft)
        ref_sig_transformed = scipy.signal.convolve(self.ref_signal, distortion_signal, mode='full')
        return ref_sig_transformed

    def run(self, ref_signal=None, seg_signal=None):
        ref_signal = self.ref_signal.copy() if ref_signal is None else ref_signal.copy()
        seg_signal = self.seg_signal.copy() if seg_signal is None else seg_signal.copy()
        max_correlation_sample, correlation = self.find_max_correlation_time(ref_signal, seg_signal)
        ref_segment_estimate, shifted_x = self.slice_estimated_segment(ref_signal, seg_signal, max_correlation_sample)
        self.ref_segment_estimate = ref_segment_estimate
        return ref_segment_estimate, shifted_x, max_correlation_sample, correlation

    def run_spectrogram(self):
        ref_spectrogram = scipy.signal.spectrogram(self.ref_signal, self.ref_sample_rate)
        seg_spectrogram = scipy.signal.spectrogram(self.seg_signal, self.seg_sample_rate)
        return ref_spectrogram, seg_spectrogram


# cross correlate
for name in dataset:
    ref_signal, seg_signal = dataset['ref']['amps'].copy().astype(float), dataset[name]['amps'].copy().astype(float)

    audio_matcher = AudioSegmentFinder(ref_signal=ref_signal,
                                       seg_signal=seg_signal,
                                       ref_sample_rate=dataset['ref']['sample_rate'],
                                       seg_sample_rate=dataset[name]['sample_rate'])

    ref_segment_estimate, shifted_x, max_correlation_sample, correlation = audio_matcher.run()
    dataset[name][
        'offset_correction'] = max_correlation_sample * audio_matcher.sample_rate_ratio  # correct sample of max correlation with sample rate ratio

    ref_spectrogram, seg_spectrogram = audio_matcher.run_spectrogram()

    # PLOTS
    # plot spectrogram
    fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(20, 10))
    fig.suptitle(name)
    axs.pcolormesh(ref_spectrogram[1], ref_spectrogram[0], ref_spectrogram[2])
    axs.set_title('Spectrogram')
    axs.set_xlabel('samples')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(20, 10))
    fig.suptitle(name)
    axs.pcolormesh(seg_spectrogram[1], seg_spectrogram[0], seg_spectrogram[2])
    axs.set_title('Spectrogram')
    axs.set_xlabel('samples')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    ref_spectrogram_img = np.zeros((len(seg_spectrogram[0]), len(seg_spectrogram[1])))
    logger.info(seg_spectrogram[2].shape)
    #spectrogram_correlation = scipy.signal.convolve2d(ref_spectrogram_img, ref_spectrogram_img)
    fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(20, 10))
    fig.suptitle(name)
    axs.pcolormesh(seg_spectrogram[1], seg_spectrogram[0], seg_spectrogram[2])
    axs.set_title('Spectrogram')
    axs.set_xlabel('samples')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(20, 10))
    fig.suptitle(name)
    #axs.imshow(spectrogram_correlation)

    #     # plot ref and segment time series signals
    #     fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(20,3))
    #     fig.suptitle(name)
    #     axs.plot(shifted_x, ref_segment_estimate, label='ref')
    #     axs.plot(shifted_x, audio_matcher.seg_signal, label=f'segment', alpha=0.7)
    #     axs.set_title('Overlap segment on ref')
    #     axs.set_xlabel('samples')
    #     axs.legend()
    #     fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    #     # plot ref and segment time series signals
    #     fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(20,3))
    #     fig.suptitle(name)
    #     axs.plot(np.abs(correlation))
    #     axs.plot([max_correlation_sample,max_correlation_sample], [np.min(correlation), np.max(correlation)], alpha=0.5)
    #     axs.set_title('Correlation')
    #     axs.set_xlabel('samples')
    #     fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    #     # Save ref signal segment for validation
    #     scipy.io.wavfile.write(os.path.join('.', 'data', 'validation', f'{name}_from_ref.wav'), dataset['ref']['sample_rate'], ref_segment_estimate.astype('int16'))

    # Print timestamp
    print(name, dataset[name]['offset_correction'] / dataset['ref']['sample_rate'], max_correlation_sample)