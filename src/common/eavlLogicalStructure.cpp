// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.

#include "eavlLogicalStructure.h"
#include "eavlLogicalStructureRegular.h"
#include "eavlLogicalStructureQuadTree.h"

eavlLogicalStructure *
eavlLogicalStructure::CreateObjFromName(const string &nm)
{
    if (nm == "eavlLogicalStructureRegular")
	return new eavlLogicalStructureRegular();
    else if (nm == "eavlLogicalStructureQuadTree")
	return new eavlLogicalStructureQuadTree();
    else
	throw;
}
