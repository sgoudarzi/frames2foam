"""Shared geometry and kinematics for the fusiform aneurysm demo.
All lengths in metres, time in seconds."""
import numpy as np
R0   = 2.0e-3      # parent artery radius (D = 4 mm)
RMAX = 5.0e-3      # sac maximum radius (D = 10 mm)
L    = 60.0e-3     # domain length
ZC   = 26.0e-3     # sac centre
H    = 9.0e-3      # sac half-length
T    = 0.8         # cardiac period (75 bpm)
EPS  = 0.08        # peak radial wall distension at sac centre (8 %)
LAG  = 0.03        # wall distension lags inflow by 30 ms
NU   = 3.3e-6      # kinematic viscosity (3.5 mPa s / 1060 kg/m3)

def s(z):
    """Sac shape function: 1 at centre, 0 with zero slope at sac ends."""
    x = (np.asarray(z) - ZC) / H
    return np.where(np.abs(x) < 1.0, np.cos(0.5*np.pi*x)**2, 0.0)

def r0(z):
    return R0 * (1.0 + (RMAX/R0 - 1.0) * s(z))

def umean(t):
    """Cross-section mean inflow velocity [m/s], carotid-like waveform."""
    tau = np.mod(t, T)
    return 0.20 + 0.45*np.exp(-((tau-0.15)/0.055)**2) + 0.08*np.exp(-((tau-0.38)/0.05)**2)

def g(t):
    """Normalised wall distension waveform, approx 0..1, lagged from inflow."""
    return (umean(t - LAG) - 0.20) / 0.45
