// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavlThresholdMutator.h"
#include "eavlCellSetExplicit.h"
#include "eavlCellSetAllPoints.h"
#include "eavlException.h"


eavlThresholdMutator::eavlThresholdMutator()
{
}


void
eavlThresholdMutator::Execute()
{
    int inCellSetIndex = dataset->GetCellSetIndex(cellsetname);
    eavlCellSet *inCells = dataset->GetCellSet(cellsetname);

    eavlField   *inField = dataset->GetField(fieldname);

    ///\todo: add nodal threshold capability
    eavlField::Association fieldAssociation = inField->GetAssociation();
	if(fieldAssociation != eavlField::ASSOC_POINTS &&
		(inField->GetAssociation() != eavlField::ASSOC_CELL_SET ||
        inField->GetAssocCellSet() != dataset->GetCellSet(inCellSetIndex)->GetName()))
    {
        THROW(eavlException,"Field for subset didn't match cell set.");
    }

    eavlArray *inArray = inField->GetArray();

    vector<int> newcells;
    int in_ncells = inCells->GetNumCells();
    for (int i=0; i<in_ncells; i++)
    {
        if (inArray->GetComponentAsDouble(i,0) >= minval &&
            inArray->GetComponentAsDouble(i,0) <= maxval)
        {
            newcells.push_back(i);
        }            
    }
    unsigned int numnewcells = newcells.size();
	
	string subsetName;
    if(fieldAssociation == eavlField::ASSOC_CELL_SET) 
    {
    	eavlExplicitConnectivity conn;
	    for (unsigned int i=0; i<numnewcells; ++i)
    	{
    	    eavlCell cell = inCells->GetCellNodes(newcells[i]);
    	    conn.AddElement(cell);
    	}
	
    	eavlCellSetExplicit *subset = new eavlCellSetExplicit(
    										string("threshold_of_")+inCells->GetName(),
                                            inCells->GetDimensionality()
                                            );
    	subset->SetCellNodeConnectivity(conn);
    	dataset->AddCellSet(subset);
    	subsetName = subset->GetName();
    }
    else 
    {
    	eavlCellSetAllPoints *subset = new eavlCellSetAllPoints(
    										string("threshold_of_")+inCells->GetName(), 
    										numnewcells
    										);
		dataset->AddCellSet(subset);
		subsetName = subset->GetName();
    }


    int numDatasetFields = dataset->GetNumFields();
    for (int i=0; i<numDatasetFields; i++)
    {
        eavlField *f = dataset->GetField(i);
        if(fieldAssociation == eavlField::ASSOC_POINTS ||
			(inField->GetAssociation() == eavlField::ASSOC_CELL_SET &&
    	     inField->GetAssocCellSet() == dataset->GetCellSet(inCellSetIndex)->GetName()))
    	{
            int numcomp = f->GetArray()->GetNumberOfComponents();
            eavlFloatArray *a = new eavlFloatArray(
                                 string("subset_of_")+f->GetArray()->GetName(),
                                 numcomp, numnewcells);
            for (unsigned int j=0; j < numnewcells; j++)
            {
                int e = newcells[j];
                for (int k=0; k<numcomp; ++k)
                    a->SetComponentFromDouble(j,k, f->GetArray()->GetComponentAsDouble(e,k));
            }

            eavlField *newfield = new eavlField(f->GetOrder(), a,
                                                fieldAssociation,
                                                subsetName);
            dataset->AddField(newfield);
        }
    }
}
