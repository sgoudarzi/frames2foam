My simulation conserved mass to 0.0004 %. The flow was still wrong.

A few weeks ago I built a small aneurysm case in OpenFOAM, with the wall moving to a prescribed motion, the way a heart model is driven by its animation frames. Every check passed. Then I plotted the outflow against the motion I had intended, and it came out as a staircase.

The solver was doing its job perfectly. It followed the wall I gave it. The problem was the wall itself. Between two frames the positions are interpolated linearly, so the wall velocity jumps every time a new frame arrives, and in a closed chamber that jump goes straight into the flow through the valves.

I could not let it go, so I measured it. At 30 frames per heartbeat, which is typical for an animation, the instantaneous flow rate was off by 24 % in an idealised left ventricle and 54 % in the aneurysm. A smooth periodic fit through the same frames brought it to about 1 %.

The part that surprised me came next. Real frames are never clean, so I added 1 % jitter. Adding more frames then made every method worse, because velocity amplifies the noise. Filtering the motion at the noise floor of the frame spectrum was the thing that actually converged.

And even after that, OpenFOAM still interpolates the boundary data linearly between the samples I wrote. At 2 ms spacing that cost another 6 % on the systolic peak. At 0.5 ms the CFD matched the fitted motion within 0.8 %.

A quick test you can do on your own case. Divide your cycle length by your frame count, and compare it with how long systolic acceleration lasts. With 30 frames over 0.8 s, you get about four frames on the steepest part of the beat.

I packaged what I built along the way as frames2foam, an open-source tool that reads OBJ or STL frames, fits the motion, warns you when the frame rate is too low, and writes the boundary data for OpenFOAM. The study and the three CFD runs are in the repository.

github.com/sgoudarzi/frames2foam

I am curious how others handle this. Do you fit the motion before it reaches the solver, or trust the frames as they come?

#CFD #OpenFOAM #CardiovascularEngineering #Biomechanics #OpenSource
