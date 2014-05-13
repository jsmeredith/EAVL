// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.

#include "eavlCellSet.h"
#include "eavlCellSetAllStructured.h"
#include "eavlCellSetExplicit.h"

eavlCellSet *
eavlCellSet::Create(const string &nm)
{
    if (nm == "eavlCellSetAllStructured")
    {
	return new eavlCellSetAllStructured();
	//	eavlRegularStructure r;
	//	return new eavlCellSetAllStructured("", r);
    }
    else if (nm == "eavlCellSetExplicit")
	return new eavlCellSetExplicit();
    else
	throw;
}

