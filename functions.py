'''
Sean Keenan, PhD Physics
Quantum Memories Group, Heriot-Watt University, Edinburgh
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
import sys
# Add the parent directory to the system path
sys.path.insert(1, r"C:\Users\keena\Documents\University\python_scripts")

from Function_files.math_functions import zoom
from Function_files.addresses import Init_Directories
dirs = Init_Directories()

import matplotlib.pyplot as mp
from numpy import abs, angle, array, conjugate, cos, exp, log, pi, sin, sqrt
from typing import Union

mp.style.use(r"C:\Users\keena\Documents\University\python_scripts\Function_files\signature.mplstyle")

class FP_characteristics:
    ''''
    A class for modeling the characteristics of a Fabry-Perot cavity
    and calculating associated parameters such as the finesse, free spectral range, 
    cavity lifetime, and PDH error signal.

    '''
    def __init__(self):
        '''
        Initializes the FP_characteristics object with default parameters.

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
        self.dir = dirs.base            # set directory
        self.folder = f'folder_name/'   # folder name
        self.fname = f'file_name'       # file name
        self.format = 'png'             # format of saved plots
        self.res = 80                   # set resolution of plots

    @property
    def absorption(self):
        '''
        Calculate effective overall distributed-loss 
        coefficient

        return : float
            effective alpha
        
        '''
        return self.alpha + (1/2*self.length) * log(1/self.r_1**2 * self.r_2**2)

    @property
    def amplitude_ref(self):
        '''
        Calculate amplitude reflectivity

        return : float
            amplitude reflection co-efficient
        
        '''
        return self.r_1**2*self.r_2**2 * exp(-2*self.absorption*self.length)
    
    @property
    def cavity_waist(self):
        '''
        Calculate the waist of the cavity

        return : float
            beam waist (m)
        
        '''
        return sqrt(self.wavelength**2*self.length*(self.curvature-self.length)/ pi**2)

    def phi(self,
            nu:Union[int, array]
            ):
        '''
        Calculate the accrued phase delay between waves 
        of different frequency in the cavity

        waves : Union[int, array]
            wavelength(s)

        return : Union[float, array]
            phase (rads)

        '''
        return (4 * pi * nu * self.length * self.n_eff) / self.c
    
    @property
    def delta_nu(self):
        '''
        compute the linewidth of a cavity
        from the lifetime

        return : float
            cavity linewidth (Hz)

        '''
        return (2 * pi * self.tau_c) ** -1
    
    def e_0(self,
              nu:Union[int, array]
              ):
        '''
        Calculate the incident complex E-field 
        
        nu : Union[float, int]
            frequency of wave (Hz)

        return : Union[int, array]
            complex incident field in the cavity

        '''
        return exp(1j*(self.phi(nu)))
    
    def e_circ(self,
                nu:Union[int, array]
                ):
        '''
        Calculate the circulating wave in the FP cavity over 
        a given wavelength range
        
        '''
        numerator = abs(self.e_inc(nu))
        denominator = abs(1 - sqrt(self.amplitude_ref)*exp(-1j*self.phi(nu)))

        return numerator/denominator
    
    def e_inc(self,
              nu:Union[int, array]
              ):
        '''
        Calculate the complex E-field of light after in 
        enters the cavity
        
        nu : Union[float, int]
            frequency of wave (Hz)

        return : Union[int, array]
            complex incident field in the cavity

        '''
        return 1 - self.r_1**2 * self.e_0(nu)

    def e_ref(self,
              nu:Union[int, array]
              ):
        '''
        Calculate the reflected complex E-field 
        
        nu : Union[float, int]
            frequency of wave (Hz)

        return : Union[int, array]
            complex reflected field 

        '''
        return self.r_1**2*exp(-1j*pi) * self.e_0(nu)
    
    def err_signal(self,
                   freqs:array,
                   detune:Union[float, int]
                   ):
        '''
        Calculate the error signal

        nu : array
            frequency of wave (Hz)
        detune : Union[float, int]
            modulation frequency (Hz)

        return : array
            the error signal
             
        '''
        f_w = self.ref_coeff(freqs)
        plus = self.ref_coeff(freqs+detune)
        minus = self.ref_coeff(freqs-detune)

        return f_w*conjugate(plus)-conjugate(f_w)*minus
    
    @property
    def finesse(self):
        '''
        Calculate cavity finesse 

        return : float
            cavity finesse
        
        '''
        return self.fsr / self.delta_nu
    
    @property
    def fsr(self):
        '''
        Calculate cavity free spectral range 

        return : float
            free-spectral range (Hz)
        
        '''
        return self.c/(2*self.length)
    
    def ref_coeff(self, 
                  nu:Union[int, array]):
        '''
        Calculate the reflection co-efficient for PDH method
        as a ratio of the total reflected and incident fields

        nu : Union[float, array]
            frequcny of incident wave (Hz)       
        '''
        phi = 2*pi*nu/self.fsr
        return self.r_1**2*(exp(1j*phi)-1)/(1 - sqrt(self.amplitude_ref)*exp(1j*phi))

    @property
    def reflectivity(self):
        '''
        Calculate geometric mean reflectivity
        
        return : float
            mean reflectivity

        '''
        return sqrt(self.r_1 * self.r_2)
    
    @property    
    def tau_c(self):
        '''
        Calculate cavity lifetime 

        '''
        return -(2*self.length)/(self.c*log(self.reflectivity**2*(1-self.alpha)**2))

    def transmitted(self,
                    nu:Union[int, array],
                    i_0:int=1,):
        '''
        Calculate the transmission for a FP cavity over 
        a given wavelength range
        
        '''
        numerator = i_0*(1-self.reflectivity*exp(-2*self.alpha*self.length))**2
        denominator = 1+self.reflectivity**2*exp(-4*self.alpha*self.length)-2*self.reflectivity*exp(-2*self.alpha*self.length)*cos(self.phi(nu)) 

        return numerator/denominator
    
    def plot_reflected(self, freqs, lims=[]):

        a, b = self.clip(freqs, lims)
        freqs = freqs[a:b]
        trans = self.transmitted(freqs)
        ref = 1 - trans

        fig, ax = mp.subplots()
        ax.plot(freqs*self.scale, ref)
        ax.set(xlabel='Detuning $\delta\\nu$', ylabel='Intensity (arb.)')

        if self.save == True:
            self.save_fig(figure = fig)

        return fig, ax

    def plot_transmitted(self, freqs, lims=[]):

        a, b = self.clip(freqs, lims)
        freqs = freqs[a:b]
        trans = self.transmitted(freqs)

        fig, ax = mp.subplots()
        ax.plot(freqs*self.scale, trans)
        ax.set(xlabel='Detuning $\delta\\nu$', ylabel='Intensity (arb.)')

        if self.save == True:
            self.save_fig(figure = fig)

        return fig, ax

    def plot_ref_coeff(self, freqs, lims=[]):

        a, b = self.clip(freqs, lims)
        freqs = freqs[a:b]
        f_w = self.ref_coeff(freqs)

        fig_1, ax_1 = mp.subplots()
        ax_1.plot(freqs*self.scale, f_w.real, label='$\Re$ F($\omega$)')
        ax_1.plot(freqs*self.scale, f_w.imag, label='$\Im$ F($\omega$)')
        ax_1.set(xlabel='Detuning $\delta\\nu$', ylabel='Intensity (arb.)')
        ax_1.legend(loc='upper left')

        ticks = [-pi, 0, pi]
        tick_labels = ['$-\pi$', 0, '$\pi$']

        fig_2, ax_2 = mp.subplots()
        ax_2.plot(freqs[freqs<0]*self.scale, angle(f_w)[freqs<0], color='C0')
        ax_2.plot(freqs[freqs>0]*self.scale, angle(f_w)[freqs>0], color='C0')
        ax_2.set(xlabel='Detuning $\delta\\nu$', ylabel='Phase')
        ax_2.set_yticks(ticks)
        ax_2.set_yticklabels(tick_labels)

        if self.save == True:
            name = self.fname
            self.fname = name + '_real'
            self.save_fig(figure = fig_1)
            self.fname = name + '_imag'
            self.save_fig(figure = fig_2)

        return [fig_1, ax_1], [fig_2, ax_2]
    
    def plot_error_sig(self, freqs, detune, lims=[]):

        a, b = self.clip(freqs, lims)
        freqs = freqs[a:b]
        error = self.err_signal(freqs, detune=detune)

        fig, ax = mp.subplots()
        ax.plot(freqs*self.scale, error.imag)
        ax.set(xlabel='Detuning $\delta\\nu$', ylabel='Intensity (arb.)')

        if self.save == True:
            self.save_fig(figure = fig)
        
        return fig, ax
    
    def save_fig(self, figure):

        self.path = f'{self.dir}{self.folder}{self.fname}.{self.format}'     # save directory
        figure.savefig(fname=self.path, dpi=self.res, format=self.format, bbox_inches='tight')

        return print('figure saved!')
    
    def clip(self, x, lims=[]):

        if lims:
            a, b = zoom(x, lims)
        else:
            a = 0
            b = None

        return a, b