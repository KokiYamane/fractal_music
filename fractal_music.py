import numpy as np
from scipy.io import wavfile

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def main():
    rate = 44100
    BPM = 240
    seconds = 60 / BPM

    def musical_scale(n):
        return 27.500 * 2 ** (n / 12)

    def make_phases(frequency, seconds, rate=rate, bias=0):
        phases = np.cumsum(
            2.0 * np.pi * frequency / rate * np.ones(int(rate * seconds)))
        phases += bias
        return phases

    def make_wave(phase):
        freq = [1, 2, 3, 4, 5, 6]
        amp = [1, 0.6, 0.4, 0.8, 0.6, 0.75]
        amp = np.array(amp) / sum(amp)
        wave = np.zeros(len(phase))
        for A, f in zip(amp, freq):
            wave += A * np.sin(f * phase)
        return wave

    melody = [np.random.randint(88) for _ in range(100)]
    freq = musical_scale(np.array(melody))

    phases_list = []
    bias = 0
    for f in freq:
        phases = make_phases(f, seconds, bias=bias)
        phases_list.append(phases)
        bias = phases[-1]
    phases = np.concatenate(phases_list)
    wave = make_wave(phases)

    # format data to int16
    wave = (wave * float(2 ** 15 - 1)).astype(np.int16)

    # save to wav file
    wavfile.write(filename='result.wav', rate=rate, data=wave)

    # plot melody
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(melody)
    ax.set_xlabel('step')
    # ax.set_ylabel('frequency [Hz]')
    fig.savefig('melody.png')

    # plot wave
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(wave)
    ax.set_xlabel('step')
    plt.xlim(rate * seconds * 0.95, rate * seconds * 1.05)
    fig.savefig('wave.png')

    # plot sample wave
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    sample_phases = np.arange(0, 2 * np.pi, 0.1)
    sample_wave = make_wave(sample_phases)
    ax.plot(sample_phases, sample_wave)
    fig.savefig('sample_wave.png')


if __name__ == '__main__':
    main()
