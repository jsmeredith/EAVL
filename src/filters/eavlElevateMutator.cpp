// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavlElevateMutator.h"
#include "eavlException.h"


eavlElevateMutator::eavlElevateMutator()
{
}


void
eavlElevateMutator::Execute()
{
    eavlField   *inField = dataset->GetField(fieldname);

    if (inField->GetAssociation() != eavlField::ASSOC_POINTS &&
        inField->GetAssociation() != eavlField::ASSOC_WHOLEMESH &&
        inField->GetAssociation() != eavlField::ASSOC_LOGICALDIM)
    {
        THROW(eavlException,"Field for elevate cannot be associated with a cell set.");
    }


    ///\todo: the fact that this is slightly less easy than it should be
    /// leads me to believe we might want to just have "cartesian" be
    /// an attribute of a single eavlCoordinates class.  Need more thought on this.
    /// e.g. it would be nice to call
    /// eavlCoordinates->AddAxis( ::Z, eavlCoordinateAxisField(blah, blah))
    /// without downcasting.  EXCEPT: what is the semantic meaning of that
    /// third dimension?  if we're in cartesian, we assume it's
    /// whichever of the dimensions we're missing.  So maybe what
    /// we REALLY want is a eavlCoordinatesCartesian::Expand() that
    /// finds the one we're missing for us!
    eavlCoordinatesCartesian *old_coords =
        dynamic_cast<eavlCoordinatesCartesian*>(dataset->GetCoordinateSystem(0));
    if (!old_coords)
        THROW(eavlException,"for the moment, assuming we've got cartesian axes in elevate....");

    eavlCoordinatesCartesian *coords;
    if (old_coords->GetDimension() == 1)
    {
        coords = new eavlCoordinatesCartesian(dataset->GetLogicalStructure(),
                                              eavlCoordinatesCartesian::X,
                                              eavlCoordinatesCartesian::Y);
        coords->SetAxis(0, old_coords->GetAxis(0));
        coords->SetAxis(1, new eavlCoordinateAxisField(inField->GetArray()->GetName()));
    }
    else if (old_coords->GetDimension() == 2)
    {
        coords = new eavlCoordinatesCartesian(dataset->GetLogicalStructure(),
                                              eavlCoordinatesCartesian::X,
                                              eavlCoordinatesCartesian::Y,
                                              eavlCoordinatesCartesian::Z);
        coords->SetAxis(0, old_coords->GetAxis(0));
        coords->SetAxis(1, old_coords->GetAxis(1));
        coords->SetAxis(2, new eavlCoordinateAxisField(inField->GetArray()->GetName()));
    }
    else
    {
        THROW(eavlException,"Can only elevate 1D or 2D coordinate systems.  "
              "(If you believe you have a 2D coordinate system but are still "
              "getting this error, it may be because your file contains, for "
              "example, Z coordinates which are all zero.  The file reader "
              "must be updated to strip this axis in that case.");
    }
    dataset->SetCoordinateSystem(0, coords);
}

