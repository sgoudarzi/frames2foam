# fourierDisplacement boundary condition

A `pointDisplacement` boundary condition that evaluates a periodic wall motion at every
solver time step from a truncated Fourier series per patch point,

    d(t) = A0 + sum_k [ A_k cos(k w (t - t0)) + B_k sin(k w (t - t0)) ],   w = 2 pi / period

with the coefficients written by `frames2foam write --format fourier`.

Compared with `timeVaryingMappedFixedValue`, the motion has no linear interpolation between
stored samples, so the wall velocity is smooth at every time step, and the case stores a few
coefficients per point instead of one sample per output time. In the aneurysm example the
coefficient file is 8.7 MB, against 333 MB of boundary data for 0.5 ms sampling.

## Build

Tested with OpenFOAM v2512 (ESI).

    source /usr/lib/openfoam/openfoam2512/etc/bashrc   # or your installation
    cd openfoam/fourierDisplacement
    wmake libso

This builds `$FOAM_USER_LIBBIN/libfourierMotion.so`.

## Use

`system/controlDict`

    libs            (fourierMotion);

`0/pointDisplacement`

    wall
    {
        type            fourierDisplacement;
        file            "constant/fourierMotion/wall";
        matchTolerance  1e-6;      // optional, relative to the patch bounding box
        rampTime        0;         // optional, smooth start over this time [s]
        value           uniform (0 0 0);
    }

Write the coefficients from the frames, for a mesh built from frame 0:

    frames2foam write "frames/heart_*.obj" --period 0.8 --case . --patch wall \
        --method spline --format fourier

`--method spline` converts the periodic spline to a Fourier series and keeps harmonics until
the wall velocity is reproduced to `--vel-tol` (default 0.1 % RMS). `--method fourier --nharm auto`
writes the noise-filtered Fourier fit directly.

## Points and parallel runs

The file stores the patch points in OpenFOAM's local patch order, so a serial case uses it
directly. If the order or count differs, as on each processor of a decomposed case, the
condition matches points by position at start-up and stops with an error if any patch point
is farther than the tolerance from the file points (wrong units or a different mesh). In the
aneurysm test, the wall motion of a two-processor run was identical to the serial run.
