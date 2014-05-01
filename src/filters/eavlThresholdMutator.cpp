// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavlThresholdMutator.h"
#include "eavlCellSetExplicit.h"
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
    if (inField->GetAssociation() != eavlField::ASSOC_CELL_SET ||
        inField->GetAssocCellSet() != dataset->GetCellSet(inCellSetIndex)->GetName())
    {
        THROW(eavlException,"Field for threshold didn't match cell set.");
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

    eavlExplicitConnectivity conn;
    for (unsigned int i=0; i<numnewcells; ++i)
    {
        eavlCell cell = inCells->GetCellNodes(newcells[i]);
        conn.AddElement(cell);
    }

    eavlCellSetExplicit *subset = new eavlCellSetExplicit(string("threshold_of_")+inCells->GetName(),
                                                          inCells->GetDimensionality());
    subset->SetCellNodeConnectivity(conn);

    int new_cellset_index = dataset->GetNumCellSets();
    dataset->AddCellSet(subset);

    for (int i=0; i<dataset->GetNumFields(); i++)
    {
        eavlField *f = dataset->GetField(i);
        if (f->GetAssociation() == eavlField::ASSOC_CELL_SET &&
            f->GetAssocCellSet() == dataset->GetCellSet(inCellSetIndex)->GetName())
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
                                                eavlField::ASSOC_CELL_SET,
                                                subset->GetName());
            dataset->AddField(newfield);
        }
    }
}
