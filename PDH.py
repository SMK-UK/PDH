'''
Sean Keenan
2024

Class for calculating the characteristics of a Fabry Perot Cavity
and calculating the PDH error signal for such a cavity.

Everything should be given in S.I. units

Equations for the cavity characteristics can be found in Fundamentals of Photonics by Saleh & Teich.
PDH equations are fully explained and derived in @DOI: 10.1119/1.1286663 by Black.

V.1.0

TODO

- adjust neff to take array of values calculated in the class

'''
import matplotlib.pyplot as mp
import numpy as np
import os
from typing import Optional, Sequence, Union

mp.style.use(r"signature.mplstyle")

class PDH:
    ''''
    A class for modelling the characteristics of a Fabry-Perot cavity
    and calculating associated parameters such as the finesse, free spectral range, 
    cavity lifetime, and PDH error signal.

    '''
    def __init__(self) -> None:
        '''
        Initializes the PDH object with default parameters.

        '''
        self.alpha = 0                  # cavity absorption co-efficient
        self.beta = 1.08                # sideband modulation depth
        self.c = 2.9972E8               # speed of light in vacuum
        self.curvature = 200E-3         # radius of curvature of the cavty mirrors
        self.length = 250E-3            # length of cavity (m)
        self.r_1 = 0.9985               # input mirror reflection
        self.r_2 = 0.9985               # output mirror reflection
        self.wavelength = 1550E-9       # design wavelength
        self.n_eff = 1                  # effective index of cavity
        self.scale = 1E-6               # change scale of x-values when plotting
        self.save = False               # choose to save plots to file
        self.dir = 'directory'         # set directory
        self.folder = 'folder_name'    # folder name
        self.fname = 'file_name'        # file name
        self.format = 'png'             # format of saved plots
        self.res = 80                   # set resolution of plots

    @property
    def absorption(self) -> float:
        '''
        Compute the effective overall distributed loss coefficient of the cavity.

        Returns
        -------
        float
            The effective absorption coefficient.
        '''
        return self.alpha + (1/2*self.length) * \
            np.log(1/self.r_1**2 * self.r_2**2)

    @property
    def amplitude_ref(self) -> float:
        '''
        Compute the amplitude reflectivity of the Fabry-Pérot cavity.

        Returns
        -------
        float
            The amplitude reflection coefficient of the cavity.
        '''
        return self.r_1**2*self.r_2**2 * \
            np.exp(-2*self.absorption*self.length)
    
    @property
    def cavity_waist(self) -> float:
        '''
        Calculate the beam waist of the cavity mode.

        Returns
        -------
        float
            Beam waist size (m).
        '''
        return np.sqrt(self.wavelength**2*self.length* \
                       (self.curvature-self.length)/ np.pi**2)

    def _clip(self,
             x:Union[np.ndarray, list],
             lims:list=[]) -> tuple:
        '''
        Finds the indices corresponding to the specified x-axis limits.

        Parameters
        ----------
        x : Union[np.ndarray, list]
            The x-axis data to be clipped.
        lims : list, optional
            A list containing the start and stop values for clipping.

        Returns
        -------
        int, int
            Indices corresponding to the start and stop values of the clipped range.
        '''
        if lims:
            a, b = self._zoom(x, bounds=lims)
        else:
            a = 0
            b = None

        return a, b

    def phi(self,
            nu:Union[int, np.ndarray]
            ) -> Union[float, np.ndarray]:
        '''
        Calculate the phase shift of the light inside the cavity.

        Parameters
        ----------
        nu : Union[int, np.ndarray]
            Frequency of the incident wave (Hz).

        Returns
        -------
        Union[float, np.ndarray]
            Phase shift in radians.
        '''
        return (4 * np.pi * nu * self.length * self.n_eff) / self.c
    
    @property
    def delta_nu(self) -> float:
        '''
        Compute the linewidth of the cavity from its lifetime.

        Returns
        -------
        float
            Cavity linewidth (Hz).
        '''
        return (2 * np.pi * self.tau_c) ** -1
    
    def e_0(self,
              nu:Union[int, np.ndarray]
              ) -> Union[complex, np.ndarray]:
        '''
        Calculate the complex incident electric field.

        Parameters
        ----------
        nu : Union[int, np.ndarray]
            Frequency of the incident wave (Hz).

        Returns
        -------
        Union[complex, np.ndarray]
            Complex incident field.
        '''
        return np.exp(1j*(self.phi(nu)))
    
    def e_circ(self,
                nu:np.ndarray
                ):
        '''
        Compute the circulating wave inside the Fabry-Pérot cavity.

        Parameters
        ----------
        nu : Union[int, np.ndarray]
            Frequency of the incident wave (Hz).

        Returns
        -------
        Union[float, np.ndarray]
            Circulating field intensity inside the cavity.
        '''
        numerator = abs(self.e_inc(nu))
        denominator = abs(1 - np.sqrt(self.amplitude_ref) \
                          * np.exp(-1j*self.phi(nu)))

        return numerator/denominator
    
    def e_inc(self,
              nu:Union[int, np.ndarray]
              ) -> Union[complex, np.ndarray]:
        '''
        Compute the complex electric field of light after entering the cavity.

        Parameters
        ----------
        nu : Union[int, np.ndarray]
            Frequency of the incident wave (Hz).

        Returns
        -------
        Union[complex, np.ndarray]
            Complex field of the incident light inside the cavity.
        '''
        return  (1-self.r_1**2) * self.e_0(nu)

    def e_ref(self,
              nu:Union[int, np.ndarray]
              ) -> Union[complex, np.ndarray]:
        '''
        Compute the reflected complex electric field from the cavity.

        Parameters
        ----------
        nu : Union[int, np.ndarray]
            Frequency of the incident wave (Hz).

        Returns
        -------
        Union[complex, np.ndarray]
            Complex reflected field.
        '''
        return self.r_1**2*np.exp(-1j*np.pi) * self.e_0(nu)
    
    def err_signal(self,
                   freqs:np.ndarray,
                   detune:Union[float, int]
                   ) -> np.ndarray:
        '''
        Compute the Pound-Drever-Hall (PDH) error signal.

        Parameters
        ----------
        freqs : np.ndarray
            Frequency values (Hz).
        detune : Union[float, int]
            Modulation frequency (Hz).

        Returns
        -------
        np.ndarray
            The PDH error signal.
        '''
        f_w = self.ref_coeff(freqs)
        plus = self.ref_coeff(freqs+detune)
        minus = self.ref_coeff(freqs-detune)

        return f_w*np.conjugate(plus)-np.conjugate(f_w)*minus
    
    @property
    def finesse(self):
        '''
        Calculate the finesse of the Fabry-Pérot cavity.

        Returns
        -------
        float
            The finesse of the cavity.
        '''
        return self.fsr / self.delta_nu
    
    @property
    def fsr(self):
        '''
        Calculate the free spectral range (FSR) of the cavity.

        Returns
        -------
        float
            Free spectral range (Hz).
        '''
        return self.c/(2*self.length)
    
    def ref_coeff(self, 
                  nu:Union[int, np.ndarray]
                  ) -> Union[complex, np.ndarray]:
        '''
        Calculate the reflection coefficient using the PDH method.

        Parameters
        ----------
        nu : Union[int, np.ndarray]
            Frequency of the incident wave (Hz).

        Returns
        -------
        Union[complex, np.ndarray]
            Reflection coefficient as a complex number.
        '''
        phi = 2*np.pi*nu/self.fsr

        return self.r_1**2*(np.exp(1j*phi)-1)/(1 - \
                            np.sqrt(self.amplitude_ref)*np.exp(1j*phi))

    @property
    def reflectivity(self) -> float:
        '''
        Compute the geometric mean reflectivity of the mirrors.

        Returns
        -------
        float
            Mean reflectivity of the cavity.
        '''
        return np.sqrt(self.r_1 * self.r_2)
    
    @property    
    def tau_c(self) -> float:
        '''
        Calculate the photon lifetime inside the cavity.

        Returns
        -------
        float
            Cavity lifetime (s).
        '''
        return -(2*self.length)/(self.c*np.log(self.reflectivity**2 \
                                               *(1-self.alpha)**2))

    def transmitted(self,
                    nu:Union[int, float, np.ndarray],
                    i_0:int=1
                    ) -> np.ndarray:
        '''
        Compute the transmitted intensity through the cavity.

        Parameters
        ----------
        nu : Union[int, np.ndarray]
            Frequency of the incident wave (Hz).
        i_0 : int, optional
            Incident intensity (default is 1).

        Returns
        -------
        Union[float, np.ndarray]
            Transmitted intensity.
        '''
        numerator = i_0*(1-self.reflectivity \
                         *np.exp(-2*self.alpha*self.length))**2
        denominator = 1+self.reflectivity**2 \
            *np.exp(-4*self.alpha*self.length)-2*self.reflectivity* \
                np.exp(-2*self.alpha*self.length)*np.cos(self.phi(nu)) 

        return numerator/denominator

    @staticmethod    
    def _zoom(data:Union[np.ndarray, list], 
              bounds:list
              ) -> tuple:
        '''
        Find the indices for zooming in on a specific range in the dataset.

        Parameters
        ----------
        data : Union[np.ndarray, list]
            The dataset to be zoomed.
        bounds : list
            A list containing the lower and upper bounds of the zoomed region.

        Returns
        -------
        int, int
            Indices corresponding to the zoomed range.
        '''
        start = np.argmin(abs(data - bounds[0]))
        stop = np.argmin(abs(data - bounds[1]))

        return start, stop
    
    def plot_reflected(self, 
                       freqs:np.ndarray, 
                       lims:Optional[Sequence[float]] = None
                       ) -> tuple[mp.Figure, mp.Axes]:
        '''
        Plot the reflected intensity as a function of frequency detuning.

        Parameters
        ----------
        freqs : np.ndarray
            Frequency values (Hz).
        lims : list, optional
            Limits for the x-axis.

        Returns
        -------
        tuple
            Figure and axis objects.
        '''
        a, b = self._clip(freqs, lims)
        freqs = freqs[a:b]
        trans = self.transmitted(freqs)
        ref = 1 - trans

        fig, ax = mp.subplots()
        ax.plot(freqs*self.scale, ref)
        ax.set(xlabel='Detuning $\\delta\\nu$', 
               ylabel='Intensity (arb.)')

        return fig, ax

    def plot_transmitted(self, 
                         freqs:np.ndarray,
                         lims:Optional[Sequence[float]] = None
                         ) -> tuple[mp.Figure, mp.Axes]:
        '''
        Plot the transmitted intensity as a function of frequency detuning.

        Parameters
        ----------
        freqs : np.ndarray
            Frequency values (Hz).
        lims : list, optional
            Limits for the x-axis.

        Returns
        -------
        tuple
            Figure and axis objects.
        '''
        a, b = self._clip(freqs, lims)
        freqs = freqs[a:b]
        trans = self.transmitted(freqs)

        fig, ax = mp.subplots()
        ax.plot(freqs*self.scale, trans)
        ax.set(xlabel='Detuning $\\delta\\nu$', ylabel='Intensity (arb.)')

        return fig, ax

    def plot_ref_coeff(self,
                       freqs:np.ndarray,
                       lims:Optional[Sequence[float]] = None
                       ) -> tuple[mp.Figure, mp.Axes]:
        '''
        Plot the real and imaginary parts of the reflection coefficient.

        Parameters
        ----------
        freqs : np.ndarray
            Frequency values (Hz).
        lims : list, optional
            Limits for the x-axis.

        Returns
        -------
        list
            List containing two figure-axis tuples.
        '''
        a, b = self._clip(freqs, lims)
        freqs = freqs[a:b]
        f_w = self.ref_coeff(freqs)

        fig_1, ax_1 = mp.subplots()
        ax_1.plot(freqs*self.scale, f_w.real, label='$\\Re$ F($\\omega$)')
        ax_1.plot(freqs*self.scale, f_w.imag, label='$\\Im$ F($\\omega$)')
        ax_1.set(xlabel='Detuning $\\delta\\nu$', ylabel='Intensity (arb.)')
        ax_1.legend(loc='upper left')

        ticks = [-np.pi, 0, np.pi]
        tick_labels = ['$-\\pi$', 0, '$\\pi$']

        fig_2, ax_2 = mp.subplots()
        ax_2.plot(freqs[freqs<0]*self.scale, np.angle(f_w)[freqs<0], color='C0')
        ax_2.plot(freqs[freqs>0]*self.scale, np.angle(f_w)[freqs>0], color='C0')
        ax_2.set(xlabel='Detuning $\\delta\\nu$', ylabel='Phase')
        ax_2.set_yticks(ticks)
        ax_2.set_yticklabels(tick_labels)

        return [fig_1, ax_1], [fig_2, ax_2]
    
    def plot_error_sig(self,
                       freqs:np.ndarray,
                       detune:float,
                       lims:Optional[Sequence[float]] = None
                       ) -> tuple[mp.Figure, mp.Axes]:
        '''
        Plot the Pound-Drever-Hall (PDH) error signal.

        Parameters
        ----------
        freqs : np.ndarray
            Frequency values (Hz).
        detune : float
            Modulation frequency (Hz).
        lims : list, optional
            Limits for the x-axis.

        Returns
        -------
        tuple
            Figure and axis objects.
        '''
        a, b = self._clip(freqs, lims)
        freqs = freqs[a:b]
        error = self.err_signal(freqs, detune=detune)

        fig, ax = mp.subplots()
        ax.plot(freqs*self.scale, error.imag)
        ax.set(xlabel='Detuning $\\delta\\nu$', ylabel='Intensity (arb.)')
        
        return fig, ax
    
    def _join(self,
              *paths
              ):
        '''
        Join multiple path segments into a single valid path.
        '''
        return os.path.normpath(os.path.join(*paths))
    
    def save_fig(self,
                 figure:mp.Figure
                 ) -> None:
        '''
        Save the generated figure to a specified directory.

        Parameters
        ----------
        figure : matplotlib.figure.Figure
            Figure object to save.

        Returns
        -------
        None
            Prints a message confirming that the figure has been saved.
        '''
        path = self._join(self.dir, self.folder, self.fname)
        figure.savefig(fname=f'{path}.{self.format}', dpi=self.res, 
                        format=self.format, bbox_inches='tight')

        return print(f'Figure saved as {path}!')