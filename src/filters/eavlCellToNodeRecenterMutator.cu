// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavlCellToNodeRecenterMutator.h"
#include "eavlDataSet.h"
#include "eavlException.h"
#include "eavlExecutor.h"
#include "eavlSourceTopologyMapOp.h"

///\todo: I think we could replace these with a 1-size-fits-all version.
struct AverageFunctor1
{
    template <class IN>
    EAVL_FUNCTOR float operator()(int shapeType, int n, int ids[], const IN inputs)
    {
        float result = 0.f;
        if (n == 0)
            return result;

        for (int i=0; i<n; i++)
            result += collect(ids[i], inputs);
        return result / float(n);
    }
};

struct AverageFunctor3
{
    template <class IN>
    EAVL_FUNCTOR tuple<float,float,float> operator()(int shapeType, int n, int ids[], const IN inputs)
    {
        tuple<float,float,float> result(0,0,0);
        if (n == 0)
            return result;

        for (int i=0; i<n; i++)
        {
            typename collecttype<IN>::const_type in(collect(ids[i], inputs));
            get<0>(result) += get<0>(in);
            get<1>(result) += get<1>(in);
            get<2>(result) += get<2>(in);
        }
        get<0>(result) /= n;
        get<1>(result) /= n;
        get<2>(result) /= n;

        return result;
    }
};


eavlCellToNodeRecenterMutator::eavlCellToNodeRecenterMutator()
{
}

void eavlCellToNodeRecenterMutator::Execute()
{
    eavlField *field = dataset->GetField(fieldname);
    eavlArray *array = field->GetArray();

    int cellSetIndex = dataset->GetCellSetIndex(cellsetname);
    eavlCellSet *cellSet = dataset->GetCellSet(cellsetname);

    if (field->GetAssociation() != eavlField::ASSOC_CELL_SET)
        THROW(eavlException, "expected cellset field")

    if (field->GetAssocCellSet() != dataset->GetCellSet(cellSetIndex)->GetName())
        THROW(eavlException, "expected cellset field")

    int npts = dataset->GetNumPoints();
    int ncomp = array->GetNumberOfComponents();

    eavlArray *result = array->Create("nodecentered_"+array->GetName(),
                                      ncomp, npts);

    if (ncomp == 1)
    {
        eavlOperation *op = 
            new_eavlSourceTopologyMapOp(cellSet,
                                        EAVL_CELLS_OF_NODES,
                                        eavlOpArgs(array),
                                        eavlOpArgs(result),
                                        AverageFunctor1());
        eavlExecutor::AddOperation(op, "1-comp recenter to the nodes");
        eavlExecutor::Go();
    }
    else if (ncomp == 3)
    {
        eavlOperation *op = 
            new_eavlSourceTopologyMapOp(cellSet,
                                        EAVL_CELLS_OF_NODES,
                                        eavlOpArgs(make_indexable(array,0),
                                                   make_indexable(array,1), 
                                                   make_indexable(array,2)),
                                        eavlOpArgs(make_indexable(result,0),
                                                   make_indexable(result,1), 
                                                   make_indexable(result,2)),
                                        AverageFunctor3());
        eavlExecutor::AddOperation(op, "3-comp recenter to the nodes");
        eavlExecutor::Go();
    }
    else
    {
        THROW(eavlException, "expected 1- or 3-component field for recenter")
    }
    dataset->AddField(new eavlField(0, result,
                                    eavlField::ASSOC_POINTS));
}
