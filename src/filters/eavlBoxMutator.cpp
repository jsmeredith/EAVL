// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavlBoxMutator.h"
#include "eavlCellSetExplicit.h"
#include "eavlException.h"


eavlBoxMutator::eavlBoxMutator()
{
}


void
eavlBoxMutator::Execute()
{
    int inCellSetIndex = dataset->GetCellSetIndex(cellsetname);
    eavlCellSet *inCells = dataset->GetCellSet(cellsetname);

    vector<int> newcells;
    int in_ncells = inCells->GetNumCells();
    eavlExplicitConnectivity conn;
    for (int i=0; i<in_ncells; i++)
    {
        eavlCell cell = inCells->GetCellNodes(i);
        bool match = true;
        for (int j=0; j<cell.numIndices; ++j)
        {
            if (dim >= 1)
            {
                double x = dataset->GetPoint(cell.indices[j], 0);
                if (x < xmin || x > xmax)
                {
                    match = false;
                    break;
                }
            }
            if (dim >= 2)
            {
                double y = dataset->GetPoint(cell.indices[j], 1);
                if (y < ymin || y > ymax)
                {
                    match = false;
                    break;
                }
            }
            if (dim >= 3)
            {
                double z = dataset->GetPoint(cell.indices[j], 2);
                if (z < zmin || z > zmax)
                {
                    match = false;
                    break;
                }
            }
        }

        if (match)
        {
            conn.AddElement(cell);
            newcells.push_back(i);
        }            
    }
    unsigned int numnewcells = conn.GetNumElements();

    eavlCellSetExplicit *subset = new eavlCellSetExplicit(string("box_of_")+inCells->GetName(),
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
