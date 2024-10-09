import pyxdf
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.decoding import SSD

raw = mne.read_epochs_eeglab('EEGDataProcess\\test.set')
raw.resample(sfreq=250)
# raw.pick_types(meg=True, ref_meg=False)
freqs_sig = 9, 12
freqs_noise = 8, 13


ssd = SSD(
    info=raw.info,
    reg="oas",
    sort_by_spectral_ratio=False,  # False for purpose of example.
    filt_params_signal=dict(
        l_freq=freqs_sig[0],
        h_freq=freqs_sig[1],
        l_trans_bandwidth=1,
        h_trans_bandwidth=1,
    ),
    filt_params_noise=dict(
        l_freq=freqs_noise[0],
        h_freq=freqs_noise[1],
        l_trans_bandwidth=1,
        h_trans_bandwidth=1,
    ),
)
ssd.fit(X=raw.get_data())

# print(f"时间范围: {raw.tmin} 秒 到 {raw.tmax} 秒")


# # print(raw)
# # raw.compute_psd(fmax=30).plot(picks="data", exclude="bads", amplitude=False)
# # raw.plot(n_channels=32)

# raw.plot_image(picks="data")
# frequecnies = np.arange(1, 15, 1)
# power = raw.compute_tfr("morlet", n_cycles=1, return_itc=False, freqs=frequecnies, decim=3, average=True)
# power.plot(combine='mean')

ssd_sources = ssd.transform(X=raw.get_data())

# pattern = mne.EvokedArray(data=ssd.patterns_[:4].T, info=ssd.info)
# pattern.plot_topomap(units=dict(mag="A.U."), time_format="")

psd, freqs = mne.time_frequency.psd_array_welch(
    ssd_sources, sfreq=raw.info["sfreq"], n_fft=500
)
# spec_ratio, sorter = ssd.get_spectral_ratio(ssd_sources)
# fig, ax = plt.subplots(1)
# ax.plot(spec_ratio, color="black")
# ax.plot(spec_ratio[sorter], color="orange", label="sorted eigenvalues")
# ax.set_xlabel("Eigenvalue Index")
# ax.set_ylabel(r"Spectral Ratio $\frac{P_f}{P_{sf}}$")
# ax.legend()
# ax.axhline(1, linestyle="--")



plt.show()
plt.pause(0)