from PDH import PDH
import numpy as np

fp = PDH()

# simple planar cavity with r_1 = r_2
fp.r_1 = 0.9
fp.r_2 = 0.9
fp.length = 1
print(round(fp.finesse, 2))

fp.scale = 1/fp.fsr                                     # set scale to give x in fsr
freqs = np.arange(-1.5*fp.fsr, 1.5*fp.fsr, fp.fsr/1e4)  # lets just plot over a few fsr
fig_r, ax_r = fp.plot_reflected(freqs, lims=[])         # plot the reflection
ax_r.set_xticks([-1, 0, 1])
ax_r.set_xticklabels(['$-\\nu_{fsr}$', '0', '$\\nu_{fsr}$'])

fig_t, ax_t = fp.plot_transmitted(freqs, lims=[])       # plot the transmission
ax_t.set_xticks([-1, 0, 1])
ax_t.set_xticklabels(['$-\\nu_{fsr}$', '0', '$\\nu_{fsr}$'])

# increase finesse for sharper error signal
fp.r_1 = 0.98
fp.r_2 = 0.98
print(round(fp.finesse, 2))

fp.scale = 1E-6         # x axis in MHz
fig_i, fig_p  = fp.plot_ref_coeff(freqs, lims=[-fp.fsr//10, fp.fsr//10])

omega = 25E6            # set detuning of modulation
fig_err, ax_err = fp.plot_error_sig(freqs, detune=omega, lims=[-omega*2, omega*2])
ax_err.set_xticks([-omega*fp.scale, 0, omega*fp.scale])
ax_err.set_xticklabels(['$-\\Omega$', '0', '$\\Omega$'])