// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavlSubsetMutator.h"
#include "eavlCellSetSubset.h"
#include "eavlException.h"


eavlSubsetMutator::eavlSubsetMutator()
{
    minval = -FLT_MAX;
    minval = +FLT_MAX;
    all_points_required = false;
}


void
eavlSubsetMutator::Execute()
{
    int inCellSetIndex = dataset->GetCellSetIndex(cellsetname);
    eavlCellSet *inCells = dataset->GetCellSet(cellsetname);
	
    eavlField   *inField = dataset->GetField(fieldname);
	
	eavlField::Association fieldAssociation = inField->GetAssociation();
	if(fieldAssociation != eavlField::ASSOC_POINTS &&
		(inField->GetAssociation() != eavlField::ASSOC_CELL_SET ||
        inField->GetAssocCellSet() != dataset->GetCellSet(inCellSetIndex)->GetName()))
    {
        THROW(eavlException,"Field for subset didn't match cell set.");
    }

    eavlArray *inArray = inField->GetArray();

    // create the subset
    eavlCellSetSubset *subset = new eavlCellSetSubset(inCells);

    subset->subset.clear();
    int in_ncells = inCells->GetNumCells();
    if (fieldAssociation == eavlField::ASSOC_CELL_SET)
    {
        for (int i=0; i<in_ncells; i++)
        {
            if (inArray->GetComponentAsDouble(i,0) >= minval &&
                inArray->GetComponentAsDouble(i,0) <= maxval)
            {
                subset->subset.push_back(i);
            }            
        }
    }
    else // (fieldAssociation == eavlField::ASSOC_POINTS)
    {
        for (int i=0; i<in_ncells; i++)
        {
            bool all_in = true;
            bool some_in = false;
            eavlCell cell = inCells->GetCellNodes(i);
            for (int j=0; j<cell.numIndices; j++)
            {
                double val = inArray->GetComponentAsDouble(cell.indices[j],0);
                if (val >= minval && val <= maxval)
                    some_in = true;
                else
                    all_in = false;
            }

            if (all_points_required)
            {
                if (all_in)
                    subset->subset.push_back(i);
            }
            else
            {
                if (some_in)
                    subset->subset.push_back(i);
            }
        }
    }
    

    //int new_cell_index = dataset->GetNumCellSets();
    dataset->AddCellSet(subset);
	
	int numDatasetFields = dataset->GetNumFields();
    for (int i=0; i<numDatasetFields; i++)
    {
        eavlField *f = dataset->GetField(i);
        if(inField->GetAssociation() == eavlField::ASSOC_CELL_SET &&
    	     inField->GetAssocCellSet() == dataset->GetCellSet(inCellSetIndex)->GetName())
    	{
            eavlFloatArray *a = new eavlFloatArray(
                                 string("subset_of_")+f->GetArray()->GetName(),
                                 f->GetArray()->GetNumberOfComponents());
            int sub_ncells = subset->GetNumCells();
            a->SetNumberOfTuples(sub_ncells);
            for (int j=0; j < sub_ncells; j++)
            {
                int e = subset->subset[j];
                a->SetComponentFromDouble(j,0, f->GetArray()->GetComponentAsDouble(e,0));
            }

            eavlField *newfield = new eavlField(f->GetOrder(), a,
                                                eavlField::ASSOC_CELL_SET,
                                                subset->GetName());
            dataset->AddField(newfield);
        }
    }
}

