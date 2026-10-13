/*---------------------------------------------------------------------------*\
    fourierDisplacement: periodic prescribed boundary displacement evaluated
    at run time from a truncated Fourier series per patch point.

    Part of frames2foam (MIT License), https://github.com/sgoudarzi/frames2foam
    Copyright (C) 2026 Sahar Goudarzi
-------------------------------------------------------------------------------
Description
    Fixed-value point patch condition for pointDisplacement:

        d(t) = A0 + sum_k [ A_k cos(k w tau) + B_k sin(k w tau) ],
        w = 2 pi / period,  tau = t - t0

    with per-point coefficient vectors written by `frames2foam write --format fourier`.
    The motion is smooth at every solver time step, so there is no linear
    interpolation between stored samples (as with timeVaryingMappedFixedValue)
    and the stored data are a few coefficients per point instead of one sample
    per output time.

    Points are matched to the patch by position when the boundary condition is
    constructed, so the same coefficient file works for serial and decomposed cases.

Usage
    In 0/pointDisplacement:
    \verbatim
    wall
    {
        type            fourierDisplacement;
        file            "constant/fourierMotion/wall";
        // optional
        matchTolerance  1e-6;      // relative to patch bounding box
        rampTime        0;         // ramp amplitude from 0 over this time [s]
        value           uniform (0 0 0);
    }
    \endverbatim
    and in system/controlDict:  libs (fourierMotion);

\*---------------------------------------------------------------------------*/

#ifndef fourierDisplacementPointPatchVectorField_H
#define fourierDisplacementPointPatchVectorField_H

#include "fixedValuePointPatchField.H"

namespace Foam
{

class fourierDisplacementPointPatchVectorField
:
    public fixedValuePointPatchField<vector>
{
    // Private data

        //- Coefficient file (relative to the global case directory)
        fileName file_;

        //- Relative tolerance for matching file points to patch points
        scalar matchTol_;

        //- Optional amplitude ramp time [s]
        scalar rampTime_;

        //- Period and phase origin [s]
        scalar period_;
        scalar t0_;

        //- Mean displacement per patch point
        vectorField A0_;

        //- Cosine and sine coefficients, one field per harmonic
        List<vectorField> A_;
        List<vectorField> B_;


    // Private member functions

        //- Read the coefficient file and map it onto this patch's points
        void readCoeffs();


public:

    TypeName("fourierDisplacement");


    // Constructors

        fourierDisplacementPointPatchVectorField
        (
            const pointPatch&,
            const DimensionedField<vector, pointMesh>&
        );

        fourierDisplacementPointPatchVectorField
        (
            const pointPatch&,
            const DimensionedField<vector, pointMesh>&,
            const dictionary&
        );

        fourierDisplacementPointPatchVectorField
        (
            const fourierDisplacementPointPatchVectorField&,
            const pointPatch&,
            const DimensionedField<vector, pointMesh>&,
            const pointPatchFieldMapper&
        );

        fourierDisplacementPointPatchVectorField
        (
            const fourierDisplacementPointPatchVectorField&,
            const DimensionedField<vector, pointMesh>&
        );

        virtual autoPtr<pointPatchField<vector>> clone() const
        {
            return pointPatchField<vector>::Clone(*this);
        }

        virtual autoPtr<pointPatchField<vector>> clone
        (
            const DimensionedField<vector, pointMesh>& iF
        ) const
        {
            return pointPatchField<vector>::Clone(*this, iF);
        }


    // Member functions

        //- Displacement at time t
        tmp<vectorField> displacement(const scalar t) const;

        virtual void updateCoeffs();

        virtual void write(Ostream&) const;
};

} // End namespace Foam

#endif
