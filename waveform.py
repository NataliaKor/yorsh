import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from datetime import datetime
import h5py
from copy import deepcopy 

from few.waveform import GenerateEMRIWaveform

from utils import *
from projectstrain import *
from integrate_backwards import *

from fastlisaresponse import ResponseWrapper
from lisatools.detector import EqualArmlengthOrbits

##################################################
# Choose which files to overwrite 
write_orbit = 0 # Overwrite orbit file
yArm_write = 0 # Overwrite projection of the waveform on the arm
instru_write = 1 # Overwrite the noise simulation
# If we do not have noise simulation we cannot create noisy data, do not forget to create a condition for this! 
# We can create noiseless TDI in two different ways

# Read config file
#config = get_config_yaml('config/ThompsonEvent.yaml')
config = get_config_yaml('config/GairEvent.yaml')
#config = get_config_yaml('config/AdamsEvent.yaml')

# Choose the type of the wavform to use
gen_wave = GenerateEMRIWaveform("FastSchwarzschildEccentricFlux", use_gpu=False)

# Define random seed
np.random.seed(14) 

# Set of parameters 
# Read from yaml configuration file 
T = config["tvec"]["Tobs"]  # years
print('T = ', T)

# !!!!!!!!!!!!!!! Figure out which dt is good. Is 10 seconds good enough or we will need 5 seconds?
dt = config["tvec"]["dt"]  # seconds
print('dt = ', dt)

M = config["default"]["M"] # Mass of the central object
print('M = ', M)
mu = config["default"]["mu"] # Mass of the compact object
print('mu = ', mu)

dist = 10.0  # Distance

costheK = np.random.uniform(low=config["limits"]["min"]["theK"], high=config["limits"]["max"]["theK"]) # PolarAngleOfSpin  #        polar angle describing the orientation of the spin-angular momentum of MBH
theK = np.arccos(costheK)
phiK = np.random.uniform(low=config["limits"]["min"]["phiK"], high=np.pi*config["limits"]["max"]["phiK"]) # AzimuthalAngleOfSpin  #      azimuthal angle ...

costheS = np.random.uniform(low=config["limits"]["min"]["theS"], high=config["limits"]["max"]["theS"])  # 0.5*np.pi - EclipticLatitude  #        polar sky localisation angle given in the solar system barycentre frame
theS = np.arccos(costheS)
phiS = np.random.uniform(low=config["limits"]["min"]["phiS"], high=np.pi*config["limits"]["max"]["phiS"]) #0.5 # EclipticLongitude  #      azimuthal solar system angle given in the solar system barycentre frame

Phi_phi0 = np.random.uniform(low=config["limits"]["min"]["Phi_phi0"], high=np.pi*config["limits"]["max"]["Phi_phi0"]) # +  initial phase in phi
Phi_theta0 = np.random.uniform(low=config["limits"]["min"]["Phi_theta0"], high=np.pi*config["limits"]["max"]["Phi_theta0"]) #    initial phase in theta
Phi_r0 = np.random.uniform(low=config["limits"]["min"]["Phi_r0"], high=np.pi*config["limits"]["max"]["Phi_r0"]) # +  initial phase in r

print('theS = ', theS)
print('phiS = ', phiS)
print('theK = ', theK)
print('phiK = ', phiK)

print('Phi_phi0 = ', Phi_phi0)
print('Phi_theta0 = ', Phi_theta0)
print('Phi_r0 = ', Phi_r0)

x0 = 1.0  #inclination angle, will be ignored in Schwarzschild waveform
a = 0.0  # spin, will be ignored in Schwarzschild waveform

# Estimate values of e0 amd p0
e_f = config["default"]["e_f"]
print('e_f = ', e_f)

# Get initial parameters from final 
e0, p0 = get_e0p0(M, mu, e_f, Phi_phi0, Phi_theta0, Phi_r0, T, dt, dist, a, x0)

print('e0 = ', e0)
print('p0 = ', p0)

# Creating h+ and hx
print('Creating h+ and h+ ...')
wave = gen_wave(
    M,
    mu,
    a,
    p0,
    e0,
    x0,
    dist,
    theS,
    phiS,
    theK,
    phiK,
    Phi_phi0,
    Phi_theta0,
    Phi_r0,
    T=T,
    dt=dt,
)

hpS, hcS = wave.real, - wave.imag

# Convert to SSB
psi = AziPol2Psi(theS, phiS, theK, phiK)
hp, hc = source2SSB(hpS, hcS, psi)

#####################################3
# Fast response to verify the results
print('Running fast response to verify the results')

sampling_frequency = 1./dt
t0 = 30000.0
# order of the langrangian interpolation
order = 25

orbits_mike = EqualArmlengthOrbits(use_gpu=False)

# 1st or 2nd or custom (see docs for custom)
tdi_gen = "2nd generation"

index_lambda = 8
index_beta = 7

tdi_kwargs_XYZ = dict(
    order=order, tdi=tdi_gen, tdi_chan="XYZ",
)

emri_lisa = ResponseWrapper(
    gen_wave,
    T,
    dt,
    index_lambda,
    index_beta,
    t0=t0,
    flip_hx=True,  # set to True if waveform is h+ - ihx
    remove_sky_coords=False,  # True if the waveform generator does not take sky coordinates
    is_ecliptic_latitude=False,  # False if using polar angle (theta)
    remove_garbage=True,  # removes the beginning of the signal that has bad information
    orbits=orbits_mike,
    use_gpu=False,
    **tdi_kwargs_XYZ,
)

XYZ = emri_lisa(
    M,
    mu,
    a,
    p0,
    e0,
    x0,
    dist,
    theS,
    phiS,
    theK,
    phiK,
    Phi_phi0,
    Phi_theta0,
    Phi_r0,
)

# Write rsults to the file
#path_fast = 'emriXYZ_dt5_05years_nonoise_kepler_tdi2_fast.h5'
#hdf5_fast = h5py.File(path_fast, 'w')
#hdf5_fast.create_dataset('X', data=XYZ[0])
#hdf5_fast.create_dataset('Y', data=XYZ[1])
#hdf5_fast.create_dataset('Z', data=XYZ[2])
#hdf5_fast.close()

# Perform fourie transfrom to calculate SNR
# TODO change to cupy
print('Calculating fft ...')
Xf_fast = np.fft.rfft(XYZ[0])
Yf_fast = np.fft.rfft(XYZ[1])
Zf_fast = np.fft.rfft(XYZ[2])

#print('XYZ[0].shape[0] = ', XYZ[0].shape[0])

freq = np.fft.rfftfreq(XYZ[0].shape[0], d=dt)

print('freq = ', freq)

# Estimate SNR of the signal computed with the fast response
noise  = "sangria"
#Af_fast = ( Zf_fast - Xf_fast )/np.sqrt(2)
#Ef_fast = ( Xf_fast - 2*Yf_fast + Zf_fast )/np.sqrt(6)
#Tf_fast = ( Xf_fast + Yf_fast + Zf_fast )/np.sqrt(3)
#snr = compute_tdi_snr_xyz(Xf_fast, Yf_fast, Zf_fast, freq) #, noise)
#print('snr = ', snr)

# Calculate SNR using LISA analysis tools
from lisatools.datacontainer import DataResidualArray
data = DataResidualArray(XYZ, dt=dt)
from lisatools.sensitivity import SensitivityMatrix, X2TDISens, Y2TDISens, Z2TDISens
sens_mat = SensitivityMatrix(data.f_arr, [X2TDISens, Y2TDISens, Z2TDISens])
from lisatools.analysiscontainer import AnalysisContainer
analysis = AnalysisContainer(data, sens_mat, signal_gen=emri_lisa)
analysis.snr()
print('snr (analysis tools) = ', analysis.snr())

plt.figure()
analysis.loglog()
plt.savefig('sens_response.png')
plt.close()

#plt.plot(np.arange(2000) * dt, h.real[:2000])
#plt.savefig('EMRI_hplus.png')


# Define parameters of the time vector
#dt = 5.
t_min = 0.
# Number of samples in orbit file
N = hp.shape[0]
t_max = N*dt  #31536000 # in sec
t = np.arange(t_min, t_max, dt)

print('N = ', N)
# Armlength
#L = 2.5e9

# Orbits from Orbits package 
# Create Keplerian orbit with the Orbits from Simulation
# Wrte it to file
# TODO create orbit at a courser grid (and then later interpolate)
from lisaorbits import KeplerianOrbits, OEMOrbits

# Some starting time and date for the orbit
# TODO change this for fixed date
# Seems to be a bug when I set t0 = a_date
a_date = datetime.today().timestamp() # variable for t0
print('a_date = ', a_date)
t0 = 0.0 #a_date #100000.0

orbits_sim = KeplerianOrbits()
if write_orbit:
    orbits_sim.write('orbit_yorsh.h5')
    #orbits_sim.write('orbit_yorsh.h5', t0=t0, dt=dt, size=N)
else:
    print('File was already written')


# Orbits from LDC
# initialise LDC orbits with the file that was produced by the sumulation package 
from ldc.lisa.orbits import Orbits
#from ldc.lisa.projection import ProjectedStrain
#from ldc.waveform.waveform import HpHc

#orbits = Orbits.type(dict({'orbit_type':'analytic', 'nominal_arm_length':2.5e9, 
#                           "initial_position": 0, "initial_rotation": 0}))
config_orbits =  {"orbit_type":'file', 'filename':'orbit_yorsh.h5'}
orbits_ldc = Orbits.type(config_orbits)


P = ProjectedStrain(orbits_ldc, theS, phiS, theK, phiK, psi, hp, hc, t)
#gwr = P.arm_response(t_min, t_max, dt, [hphc])
yArm = P.arm_response(t_min, t_max, dt)

path_fast = 'project.h5'
hdf5_fast = h5py.File(path_fast, 'w')
hdf5_fast.create_dataset('y0', data=yArm[:,0])
hdf5_fast.create_dataset('y1', data=yArm[:,1])
hdf5_fast.create_dataset('y2', data=yArm[:,2])
hdf5_fast.create_dataset('y3', data=yArm[:,3])
hdf5_fast.create_dataset('y4', data=yArm[:,4])
hdf5_fast.create_dataset('y5', data=yArm[:,5])
hdf5_fast.close()



print('yArm.shape = ', yArm.shape)


# define time vector to write down to the file
#t = np.arange(t_min, t_max, dt) 

print('yArm.shape = ', yArm.shape)
print('t.shape = ', t.shape)
#plt.plot(np.arange(2000)*dt, yArm[:2000,0])
plt.plot(np.arange(N)*dt, yArm[:,0])
plt.savefig('arm1_all.png')
plt.close()


# Convert the projected strain to the lisagwresponse package
from lisagwresponse import ReadResponse
# Order of the variable t_interp, y_12, y_23, y_31, y_13, y_32, y_21
simlen = N
yArm_sim = ReadResponse(t[-simlen:], yArm[-simlen:,0], yArm[-simlen:,1], yArm[-simlen:,2], yArm[-simlen:,3], yArm[-simlen:,4], yArm[-simlen:,5], orbits='orbit_yorsh.h5') #, gw_beta=np.pi / 4, gw_lambda=0, dt=1, size=100)
# Write to file
print('t.shape[0] = ', t.shape[0])
if yArm_write:
     yArm_sim.write("emri_yorsh.h5", t0=t0, dt=dt, size=simlen, mode='w')
else:
     print('Already written the response to the file')

# Plot what has been ported to the gw_response
yArm_sim.plot(t[-simlen:])
plt.savefig('yArm_sim.png')

# Calculate pyTDI
from pytdi import Data
from pytdi.michelson import X2, Y2, Z2

# Simulate no noise data in the other way
data_nonoise = Data.from_gws('emri_yorsh.h5', 'orbit_yorsh.h5')
X_nonoise = X2.build(**data_nonoise.args)(data_nonoise.measurements) 
Y_nonoise = Y2.build(**data_nonoise.args)(data_nonoise.measurements) 
Z_nonoise = Z2.build(**data_nonoise.args)(data_nonoise.measurements) 

# Perform fourie transfrom to calculate SNR
# TODO fix fft with cupy on cluster
Xf = np.fft.rfft(X_nonoise)
Yf = np.fft.rfft(Y_nonoise)
Zf = np.fft.rfft(Z_nonoise)

freq = np.fft.rfftfreq(N, d=dt)

# Estimate SNR of the signal
from utils import *
noise  = "sangria"
snr = compute_tdi_snr_xyz(Xf, Yf, Zf, freq) #, noise)
print('snr = ', snr)

skip = 300

fig, ax = plt.subplots(3,1, figsize = (12,9), sharex = True)
ax[0].plot(X_nonoise[skip:], label="TDI X")
ax[1].plot(Y_nonoise[skip:], label="TDI Y")
ax[2].plot(Z_nonoise[skip:], label="TDI Z")
ax[2].set_xlabel("Time [s]")
ax[0].set_ylabel("TDI"), ax[1].set_ylabel("TDI"), ax[2].set_ylabel("TDI")
ax[0].legend(), ax[1].legend(), ax[2].legend()
plt.savefig('TDIs_nonoise.png')
plt.close()

path = 'emriXYZ_dt5_05years_nonoise_kepler_tdi2.h5'
hdf5 = h5py.File(path, 'w')
hdf5.create_dataset('X', data=X_nonoise)
hdf5.create_dataset('Y', data=Y_nonoise)
hdf5.create_dataset('Z', data=Z_nonoise)
hdf5.close()
               

# Way to get TDI and include the instrument
# Use LISA Instrument
from lisainstrument import Instrument

# TODO what if I remove aafilter=None
#instru = Instrument(aafilter=None, t0=t0, dt=dt, size=(simlen - 500000), orbits = 'orbit_yorsh.h5', gws = 'emri_yorsh.h5', lock='six')
instru = Instrument(aafilter=None, t0=t0, dt=dt, size=simlen, orbits = 'orbit_yorsh.h5', gws = 'emri_yorsh.h5', lock='six')
instru.disable_all_noises(excluding=['laser', 'test-mass', 'oms'])
instru.simulate()
instru_write = 1
if instru_write:
    instru.write('simulation_noise_emri.h5', mode = 'w')
else:
    print('Already written the instrument output to the file')

# Here we can also have data without the noise if we switch off the noise
simdata = Data.from_instrument('simulation_noise_emri.h5')

X = X2.build(**simdata.args)(simdata.measurements) / instru.central_freq
Y = Y2.build(**simdata.args)(simdata.measurements) / instru.central_freq
Z = Z2.build(**simdata.args)(simdata.measurements) / instru.central_freq

skip = 300

fig, ax = plt.subplots(3,1, figsize = (12,9), sharex = True)
ax[0].plot(instru.t[skip:], X[skip:], label="TDI X")
ax[1].plot(instru.t[skip:], Y[skip:], label="TDI Y")
ax[2].plot(instru.t[skip:], Z[skip:], label="TDI Z")
ax[2].set_xlabel("Time [s]")
ax[0].set_ylabel("TDI"), ax[1].set_ylabel("TDI"), ax[2].set_ylabel("TDI")
ax[0].legend(), ax[1].legend(), ax[2].legend()
plt.savefig('TDIs.png')
plt.close()




#path = 'emriXYZ_dt5_02years_nonoise_kepler_tdi2.h5'

#hdf5 = h5py.File(path, 'w')
#hdf5.create_dataset('X', data=X_nonoise)
#hdf5.create_dataset('Y', data=Y_nonoise)
#hdf5.create_dataset('Z', data=Z_nonoise)
#hdf5.close()
