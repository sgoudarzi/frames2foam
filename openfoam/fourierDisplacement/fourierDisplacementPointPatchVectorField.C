/*---------------------------------------------------------------------------*\
    Part of frames2foam (MIT License), https://github.com/sgoudarzi/frames2foam
    Copyright (C) 2026 Sahar Goudarzi
\*---------------------------------------------------------------------------*/

#include "fourierDisplacementPointPatchVectorField.H"
#include "pointPatchFields.H"
#include "addToRunTimeSelectionTable.H"
#include "Time.H"
#include "polyMesh.H"
#include "IFstream.H"
#include "mathematicalConstants.H"

namespace Foam
{

// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

void fourierDisplacementPointPatchVectorField::readCoeffs()
{
    const polyMesh& mesh = this->internalField().mesh()();
    const Time& runTime = mesh.time();

    fileName path(file_);
    path.expand();
    if (!path.isAbsolute())
    {
        path = runTime.globalPath()/path;
    }

    IFstream is(path);
    if (!is.good())
    {
        FatalErrorInFunction
            << "Cannot open coefficient file " << path
            << " for patch " << this->patch().name() << exit(FatalError);
    }
    const dictionary dict(is);

    period_ = dict.get<scalar>("period");
    t0_ = dict.getOrDefault<scalar>("t0", 0);

    const pointField filePoints(dict.get<pointField>("points"));
    const vectorField fileA0(dict.get<vectorField>("A0"));
    const List<vectorField> fileA(dict.get<List<vectorField>>("A"));
    const List<vectorField> fileB(dict.get<List<vectorField>>("B"));

    const label nh = fileA.size();
    if (fileB.size() != nh || fileA0.size() != filePoints.size())
    {
        FatalErrorInFunction
            << "Inconsistent sizes in " << path << exit(FatalError);
    }
    forAll(fileA, k)
    {
        if (fileA[k].size() != filePoints.size() || fileB[k].size() != filePoints.size())
        {
            FatalErrorInFunction
                << "Harmonic " << k + 1 << " has the wrong number of points in "
                << path << exit(FatalError);
        }
    }

    // Map file points onto the patch points (reference geometry)
    const pointField& pp = this->patch().localPoints();
    const label np = pp.size();

    scalar span = SMALL;
    if (np)
    {
        const boundBox bb(pp, false);
        span = max(bb.mag(), SMALL);
    }
    const scalar tol = matchTol_*span;

    labelList addr(np, -1);
    bool ordered = (filePoints.size() == np);
    if (ordered)
    {
        forAll(pp, i)
        {
            if (mag(pp[i] - filePoints[i]) > tol)
            {
                ordered = false;
                break;
            }
        }
    }

    scalar maxDist = 0;
    if (ordered)
    {
        forAll(addr, i) { addr[i] = i; }
    }
    else
    {
        // Brute-force nearest point: one-off cost at start-up, also covers
        // the subset of points owned by a processor in a decomposed case
        forAll(pp, i)
        {
            scalar best = GREAT;
            label bestj = -1;
            forAll(filePoints, j)
            {
                const scalar d = magSqr(pp[i] - filePoints[j]);
                if (d < best)
                {
                    best = d;
                    bestj = j;
                }
            }
            addr[i] = bestj;
            maxDist = max(maxDist, Foam::sqrt(best));
        }
        if (maxDist > tol)
        {
            FatalErrorInFunction
                << "Patch " << this->patch().name() << ": a patch point lies "
                << maxDist << " from the nearest point in " << path
                << " (tolerance " << tol << "). The coefficients were written for a "
                << "different mesh, or in different units." << exit(FatalError);
        }
    }

    A0_ = vectorField(fileA0, addr);
    A_.setSize(nh);
    B_.setSize(nh);
    forAll(fileA, k)
    {
        A_[k] = vectorField(fileA[k], addr);
        B_[k] = vectorField(fileB[k], addr);
    }

    Info<< "fourierDisplacement: patch " << this->patch().name()
        << ", " << np << " points, " << nh << " harmonics, period "
        << period_ << " s" << (ordered ? "" : " (matched by position)")
        << endl;
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

fourierDisplacementPointPatchVectorField::
fourierDisplacementPointPatchVectorField
(
    const pointPatch& p,
    const DimensionedField<vector, pointMesh>& iF
)
:
    fixedValuePointPatchField<vector>(p, iF),
    file_(),
    matchTol_(1e-6),
    rampTime_(0),
    period_(1),
    t0_(0),
    A0_(p.size(), Zero),
    A_(),
    B_()
{}


fourierDisplacementPointPatchVectorField::
fourierDisplacementPointPatchVectorField
(
    const pointPatch& p,
    const DimensionedField<vector, pointMesh>& iF,
    const dictionary& dict
)
:
    fixedValuePointPatchField<vector>(p, iF, dict, IOobjectOption::LAZY_READ),
    file_(dict.get<fileName>("file")),
    matchTol_(dict.getOrDefault<scalar>("matchTolerance", 1e-6)),
    rampTime_(dict.getOrDefault<scalar>("rampTime", 0)),
    period_(1),
    t0_(0),
    A0_(),
    A_(),
    B_()
{
    readCoeffs();

    if (!this->readValueEntry(dict))
    {
        Field<vector>::operator=
        (
            displacement(this->internalField().mesh()().time().value())
        );
    }
}


fourierDisplacementPointPatchVectorField::
fourierDisplacementPointPatchVectorField
(
    const fourierDisplacementPointPatchVectorField& ptf,
    const pointPatch& p,
    const DimensionedField<vector, pointMesh>& iF,
    const pointPatchFieldMapper& mapper
)
:
    fixedValuePointPatchField<vector>(ptf, p, iF, mapper),
    file_(ptf.file_),
    matchTol_(ptf.matchTol_),
    rampTime_(ptf.rampTime_),
    period_(ptf.period_),
    t0_(ptf.t0_),
    A0_(ptf.A0_, mapper),
    A_(ptf.A_.size()),
    B_(ptf.B_.size())
{
    // Map the coefficients like any point field (used by decomposePar and
    // reconstructPar). When the case is read again, the dictionary constructor
    // re-reads the file and matches points by position.
    forAll(A_, k)
    {
        A_[k] = vectorField(ptf.A_[k], mapper);
        B_[k] = vectorField(ptf.B_[k], mapper);
    }
}


fourierDisplacementPointPatchVectorField::
fourierDisplacementPointPatchVectorField
(
    const fourierDisplacementPointPatchVectorField& ptf,
    const DimensionedField<vector, pointMesh>& iF
)
:
    fixedValuePointPatchField<vector>(ptf, iF),
    file_(ptf.file_),
    matchTol_(ptf.matchTol_),
    rampTime_(ptf.rampTime_),
    period_(ptf.period_),
    t0_(ptf.t0_),
    A0_(ptf.A0_),
    A_(ptf.A_),
    B_(ptf.B_)
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

tmp<vectorField>
fourierDisplacementPointPatchVectorField::displacement(const scalar t) const
{
    const scalar w = constant::mathematical::twoPi/period_;
    const scalar tau = t - t0_;

    auto td = tmp<vectorField>::New(A0_);
    vectorField& d = td.ref();

    forAll(A_, k)
    {
        const scalar phase = (k + 1)*w*tau;
        d += Foam::cos(phase)*A_[k] + Foam::sin(phase)*B_[k];
    }

    if (rampTime_ > 0 && t < rampTime_)
    {
        const scalar r = max(t, 0)/rampTime_;
        d *= r*r*(3 - 2*r);                       // smoothstep
    }

    return td;
}


void fourierDisplacementPointPatchVectorField::updateCoeffs()
{
    if (this->updated())
    {
        return;
    }

    const Time& t = this->internalField().mesh()().time();
    Field<vector>::operator=(displacement(t.value()));

    fixedValuePointPatchField<vector>::updateCoeffs();
}


void fourierDisplacementPointPatchVectorField::write(Ostream& os) const
{
    pointPatchField<vector>::write(os);
    os.writeEntry("file", file_);
    os.writeEntryIfDifferent<scalar>("matchTolerance", 1e-6, matchTol_);
    os.writeEntryIfDifferent<scalar>("rampTime", 0, rampTime_);
    this->writeValueEntry(os);
}


makePointPatchTypeField
(
    pointPatchVectorField,
    fourierDisplacementPointPatchVectorField
);

} // End namespace Foam
