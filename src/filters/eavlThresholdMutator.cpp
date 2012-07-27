// Copyright 2010-2012 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavlThresholdMutator.h"
#include "eavlCellSetSubset.h"
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

    if (inField->GetAssociation() != eavlField::ASSOC_CELL_SET ||
        inField->GetAssocCellSet() != inCellSetIndex)
    {
        THROW(eavlException,"Field for threshold didn't match cell set.");
    }


    eavlArray *inArray = inField->GetArray();

    // create the subset
    eavlCellSetSubset *subset = new eavlCellSetSubset(inCells);

    subset->subset.clear();
    int in_ncells = inCells->GetNumCells();
    for (int i=0; i<in_ncells; i++)
    {
        if (inArray->GetComponentAsDouble(i,0) >= minval &&
            inArray->GetComponentAsDouble(i,0) <= maxval)
        {
            subset->subset.push_back(i);
        }            
    }

    int new_cell_index = dataset->cellsets.size();
    dataset->cellsets.push_back(subset);

    for (int i=0; i<dataset->fields.size(); i++)
    {
        eavlField *f = dataset->fields[i];
        if (f->GetAssociation() == eavlField::ASSOC_CELL_SET &&
            f->GetAssocCellSet() == inCellSetIndex)
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
                                                new_cell_index);
            dataset->fields.push_back(newfield);
        }
    }
}
