#include "eavlCoordinates.h"

eavlCoordinateAxis*
eavlCoordinateAxis::Create(const string &nm)
{
    if (nm == "eavlCoordinateAxisField")
	return new eavlCoordinateAxisField();
    else if (nm == "eavlCoordinateAxisRegular")
	return new eavlCoordinateAxisRegular();
}

eavlCoordinates*
eavlCoordinates::Create(const string &nm)
{
    if (nm == "eavlCoordinatesCartesian")
	return new eavlCoordinatesCartesian();
    else if (nm == "eavlCoordinatesCartesianWithTransform")
	return new eavlCoordinatesCartesianWithTransform();
    else
	throw;
}
