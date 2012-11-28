#include "eavlCellToNodeRecenterMutator.h"
#include "eavlDataSet.h"
#include "eavlException.h"
#include "eavlExecutor.h"
#include "eavlTopologyMapOp_1_0_1.h"
#include "eavlTopologyMapOp_3_0_3.h"

struct AverageFunctor1
{
    EAVL_FUNCTOR float operator()(int shapeType, int n, float vals[])
    {
        float result = 0.f;
        for (int i=0; i<n; i++)
            result += vals[i];
        return result / float(n);
    }
};

struct AverageFunctor3
{
    EAVL_FUNCTOR void operator()(int shapeType, int n,
                                 float ivals0[], float ivals1[], float ivals2[],
                                 float &oval0, float &oval1, float &oval2)
    {
        float result0 = 0.f;
        float result1 = 0.f;
        float result2 = 0.f;
        for (int i=0; i<n; i++)
        {
            result0 += ivals0[i];
            result1 += ivals1[i];
            result2 += ivals2[i];
        }
        oval0 = result0 / float(n);
        oval1 = result1 / float(n);
        oval2 = result2 / float(n);
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
        THROW(eavlException, "expected nodal field")

    if (field->GetAssocCellSet() != cellSetIndex)
        THROW(eavlException, "expected nodal field")

    int ncells = cellSet->GetNumCells();
    int npts = dataset->GetNumPoints();
    int ncomp = array->GetNumberOfComponents();

    eavlArray *result = array->Create("nodecentered_"+array->GetName(),
                                      ncomp, npts);

    if (ncomp == 1)
    {
        eavlOperation *op =
            new eavlTopologyMapOp_1_0_1<AverageFunctor1>(cellSet,
                                                        EAVL_CELLS_OF_NODES,
                                                        array,
                                                        result,
                                                        AverageFunctor1());

        eavlExecutor::AddOperation(op, "recenter to the cells");
        eavlExecutor::Go();
    }
    else if (ncomp == 3)
    {
        eavlOperation *op =
            new eavlTopologyMapOp_3_0_3<AverageFunctor3>(cellSet,
                                                        EAVL_CELLS_OF_NODES,
                                                        eavlArrayWithLinearIndex(array,0),
                                                        eavlArrayWithLinearIndex(array,1),
                                                        eavlArrayWithLinearIndex(array,2),
                                                        eavlArrayWithLinearIndex(result,0),
                                                        eavlArrayWithLinearIndex(result,1),
                                                        eavlArrayWithLinearIndex(result,2),
                                                        AverageFunctor3());

        eavlExecutor::AddOperation(op, "recenter to the cells");
        eavlExecutor::Go();
    }
    else
    {
        THROW(eavlException, "expected 1- or 3-component field for recenter")
    }
    dataset->AddField(new eavlField(0, result,
                                    eavlField::ASSOC_POINTS));
}
