#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2017-2020, Node Supply Chain Manager Corporation Limited.
@contact: lionhe0119@hotmail.com
@file: FBank.py
@time: 2020/10/12 11:05 下午
@desc:
"""
import soundfile
import os
import numpy as np

from experiments.AudioPreprocess.audio import AudioSegment


def fbank(filename):
    segment = AudioSegment.from_file(filename)
    segment.normalize(target_db=-20)
    a = compute_linear_specgram(segment.samples, segment.sample_rate)
    return a


def compute_linear_specgram(samples,
                            sample_rate,
                            stride_ms=15.0,
                            window_ms=25.0,
                            max_freq=None,
                            eps=1e-14):
    """Compute the linear spectrogram from FFT energy."""
    if max_freq is None:
        max_freq = sample_rate / 2
    if max_freq > sample_rate / 2:
        raise ValueError("max_freq must not be greater than half of "
                         "sample rate.")
    if stride_ms > window_ms:
        raise ValueError("Stride size must not be greater than "
                         "window size.")
    stride_size = int(0.001 * sample_rate * stride_ms)
    window_size = int(0.001 * sample_rate * window_ms)
    specgram, freqs = specgram_real(
        samples,
        window_size=window_size,
        stride_size=stride_size,
        sample_rate=sample_rate)
    ind = np.where(freqs <= max_freq)[0][-1] + 1
    return np.log(specgram[:ind, :] + eps)


def specgram_real(samples, window_size, stride_size, sample_rate=16000):
    truncate_size = (len(samples) - window_size) % stride_size
    samples = samples[:len(samples) - truncate_size]
    nshape = (window_size, (len(samples) - window_size) // stride_size + 1)
    nstrides = (samples.strides[0], samples.strides[0] * stride_size)
    windows = np.lib.stride_tricks.as_strided(
        samples, shape=nshape, strides=nstrides)
    assert np.all(windows[:, 1] == samples[stride_size:(stride_size + window_size)])
    # window weighting, squared Fast Fourier Transform (fft), scaling
    weighting = np.hanning(window_size)[:, None]
    fft = np.fft.rfft(windows * weighting, axis=0)
    fft = np.absolute(fft)
    fft = fft ** 2
    scale = np.sum(weighting ** 2) * sample_rate
    fft[1:-1, :] *= (2.0 / scale)
    fft[(0, -1), :] /= scale
    # prepare fft frequency list
    freqs = float(sample_rate) / window_size * np.arange(fft.shape[0])
    return fft, freqs


if __name__ == '__main__':
    dirname = "/Users/coffee/Documents/dateset/data_speech_commands_v0.02"
    filename = os.path.join(dirname, "four/c9e251d2_nohash_1.wav")
    fbank(filename)
