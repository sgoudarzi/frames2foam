"""Periodic temporal fitting of vertex trajectories."""
import numpy as np
from scipy.interpolate import CubicSpline

def harmonic_power(c):
    """Power per harmonic, summed over vertices and components ([nk])."""
    return (np.abs(c) ** 2).reshape(len(c), -1).sum(1)

def auto_harmonics(c, factor=10.0):
    """Choose the number of harmonics to keep from the frame spectrum.

    Smooth motion has a spectrum that keeps falling until round-off; frame jitter
    flattens the tail into a white noise floor. The floor is estimated as the median
    power of the upper half of the resolved harmonics. If the spectrum is still falling
    steeply across that band the data are treated as clean and every harmonic is kept;
    otherwise harmonics are kept up to the last one exceeding factor x floor."""
    P = harmonic_power(c)[1:]                          # exclude the mean
    n = len(P)
    if n < 6:
        return n
    tail = P[n // 2:]
    floor = np.median(tail)
    q = max(len(tail) // 4, 1)
    falling = np.median(tail[:q]) > 30.0 * np.median(tail[-q:])
    if falling or floor <= 1e-12 * P.max():
        return n
    above = np.nonzero(P > factor * floor)[0]
    return int(above[-1] + 1) if len(above) else 1

class PeriodicMotion:
    """Fit a periodic motion through frames sampled uniformly over one period.

    X       : array [nframes, nv, 3] of vertex positions, frame k at t = t0 + k*period/nframes.
              If the last frame repeats the first (closed loop), it is dropped automatically.
    period  : cycle duration [s]
    method  : 'spline'  periodic cubic spline (default)
              'fourier' trigonometric interpolation; with nharm < nframes/2 a low-pass fit,
                        nharm='auto' picks the cut-off from the frame spectrum (noise floor)
              'linear'  piecewise linear (what most pipelines do implicitly)
    """
    def __init__(self, X, period, method='spline', nharm=None, t0=0.0, closed_tol=1e-9):
        X = np.asarray(X, float)
        span = np.ptp(X[0], axis=0).max() if X.shape[1] else 1.0
        if len(X) > 2 and np.abs(X[-1] - X[0]).max() <= closed_tol * max(span, 1e-30):
            X = X[:-1]
        self.X, self.T, self.t0, self.method = X, float(period), float(t0), method
        self.n = len(X)
        self.dt = self.T / self.n
        self.tk = self.t0 + self.dt * np.arange(self.n)
        if method == 'spline':
            self._sp = CubicSpline(np.r_[self.tk, self.t0 + self.T], np.concatenate([X, X[:1]]),
                                   bc_type='periodic', axis=0)
        elif method == 'fourier':
            self._c = np.fft.rfft(X, axis=0) / self.n
            kmax = self.n // 2
            if nharm == 'auto':
                self.nharm = auto_harmonics(self._c)
            else:
                self.nharm = kmax if nharm is None else min(int(nharm), kmax)
            self._k = np.arange(self.nharm + 1)
            w = np.ones(self.nharm + 1); w[1:] = 2.0
            if self.n % 2 == 0 and self.nharm == kmax:
                w[-1] = 1.0                                  # Nyquist term counted once
            self._w = w
        elif method != 'linear':
            raise ValueError("method must be 'spline', 'fourier' or 'linear'")

    def _phase(self, t):
        return np.mod(np.asarray(t, float) - self.t0, self.T)

    def position(self, t):
        tau = self._phase(t)
        if self.method == 'spline':
            return self._sp(self.t0 + tau)
        if self.method == 'fourier':
            e = np.exp(2j * np.pi * np.multiply.outer(tau, self._k) / self.T) * self._w
            return np.real(np.tensordot(e, self._c[:self.nharm + 1], axes=(-1, 0)))
        s = tau / self.dt; i0 = np.floor(s).astype(int) % self.n; a = (s - np.floor(s))
        i1 = (i0 + 1) % self.n
        a = np.expand_dims(a, (-1, -2))
        return (1 - a) * self.X[i0] + a * self.X[i1]

    def velocity(self, t):
        tau = self._phase(t)
        if self.method == 'spline':
            return self._sp(self.t0 + tau, 1)
        if self.method == 'fourier':
            k = self._k
            e = (2j * np.pi * k / self.T) * np.exp(2j * np.pi * np.multiply.outer(tau, k) / self.T) * self._w
            return np.real(np.tensordot(e, self._c[:self.nharm + 1], axes=(-1, 0)))
        s = tau / self.dt; i0 = np.floor(s).astype(int) % self.n; i1 = (i0 + 1) % self.n
        return (self.X[i1] - self.X[i0]) / self.dt
