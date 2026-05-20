"""Physical and numerical constants for bigspy."""

# Speed of light (km/s)
C_LIGHT = 299792.458

# Log-wavelength spacing of templates
DLOGW = 0.0001

# Velocity scale per pixel (km/s) for log-wavelength grid
DLOGW_VEL = (10 ** DLOGW - 1) * C_LIGHT

# Number of PCA components available
NEIG = 20

# Number of PCA components to use in fit
FIT_NEIG = 10

# Normalization wavelength (Angstrom)
WAVE_NORM = 5500.0
