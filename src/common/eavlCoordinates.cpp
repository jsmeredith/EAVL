#include "eavlCoordinates.h"

eavlCoordinateAxis*
eavlCoordinateAxis::CreateObjFromName(const string &nm)
{
    if (nm == "eavlCoordinateAxisField")
	return new eavlCoordinateAxisField();
    else if (nm == "eavlCoordinateAxisRegular")
	return new eavlCoordinateAxisRegular();
    else
	throw;
}

eavlCoordinates*
eavlCoordinates::CreateObjFromName(const string &nm)
{
    if (nm == "eavlCoordinatesCartesian")
	return new eavlCoordinatesCartesian();
    else if (nm == "eavlCoordinatesCartesianWithTransform")
	return new eavlCoordinatesCartesianWithTransform();
    else
	throw;
}
