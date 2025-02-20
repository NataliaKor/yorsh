from ldc.lisa.noise import Noise, AnalyticNoise
import numpy as np

from astropy.cosmology import FlatLambdaCDM, z_at_value
import astropy.units as u

from ruamel.yaml import YAML

def get_config_yaml(path):
    """
    Read configuration parameter form file
    Args: 
        path -- Path to the yaml configuration file
    Return: 
        Dict with parameter
    """
    yaml = YAML()
    with open(path, 'r') as stream:
        return yaml.load(stream)


# Convert red shift to distance
def DL(z):

    ldc_cosmo = FlatLambdaCDM(H0=67.1, Om0=0.3175)
    quantity = ldc_cosmo.luminosity_distance(z)
    return quantity.value, quantity.unit

def z_from_dist(d):
    
    ldc_cosmo = FlatLambdaCDM(H0=67.1, Om0=0.3175)
    return z_at_value(ldc_cosmo, d * u.Gpc).value

# TODO: check if the noise is correct
# Compute SNR
#
#def compute_tdi_snr_xyz(X, Y, Z, freq): #, noise):
#    """ Compute SNR from TDI X,Y,Z.
#    """
#    source = {"X":X, "Y":Y, "Z":Z}
#    df = freq[2] - freq[1]
#
#    noise = AnalyticNoise(freq, model="sangria")
#    SXX = noise.psd(freq=freq, option='X', tdi2=True) # noise["X"].sel(f=freq, method="nearest").values
#    
#
#    snr_tot = 0  # total SNR
#    for k in ["X", "Y", "Z"]:
#        s = source[k]#data[k].sel(f=freq, method="nearest").values
#
#        snr = np.nansum(np.real(s*np.conj(s)/SXX))
#        snr *= 4.0*df
#        snr_tot += snr
#
#    return snr_tot


