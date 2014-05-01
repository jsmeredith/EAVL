// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavlSubsetMutator.h"
#include "eavlCellSetSubset.h"
#include "eavlException.h"


eavlSubsetMutator::eavlSubsetMutator()
{
}


void
eavlSubsetMutator::Execute()
{
    int inCellSetIndex = dataset->GetCellSetIndex(cellsetname);
    eavlCellSet *inCells = dataset->GetCellSet(cellsetname);

    eavlField   *inField = dataset->GetField(fieldname);

    if (inField->GetAssociation() != eavlField::ASSOC_CELL_SET ||
        inField->GetAssocCellSet() != dataset->GetCellSet(inCellSetIndex)->GetName())
    {
        THROW(eavlException,"Field for subset didn't match cell set.");
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

    int new_cell_index = dataset->GetNumCellSets();
    dataset->AddCellSet(subset);

    for (int i=0; i<dataset->GetNumFields(); i++)
    {
        eavlField *f = dataset->GetField(i);
        if (f->GetAssociation() == eavlField::ASSOC_CELL_SET &&
            f->GetAssocCellSet() == dataset->GetCellSet(inCellSetIndex)->GetName())
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
