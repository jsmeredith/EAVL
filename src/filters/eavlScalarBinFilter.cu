// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavlScalarBinFilter.h"
#include "eavlDataSet.h"
#include "eavlException.h"
#include "eavlExecutor.h"
#include "eavlCellSetAllStructured.h"
#include "eavlReduceOp_1.h"
#include "eavlMapOp.h"

struct InRange
{
    float lo, hi;
    InRange(float l, float h) : lo(l), hi(h) { }
    EAVL_FUNCTOR int operator()(float val)
    {
        return (val >= lo) && (val < hi);
    }
};

eavlScalarBinFilter::eavlScalarBinFilter()
{
}

void eavlScalarBinFilter::Execute()
{
    eavlField *field = input->GetField(fieldname);
    eavlArray *array = field->GetArray();

    if (array->GetNumberOfComponents() != 1)
        THROW(eavlException, "expected single-component field");

    eavlArray *minval = array->Create("minval", 1, 1);
    eavlArray *maxval = array->Create("maxval", 1, 1);

    eavlExecutor::AddOperation(
        new eavlReduceOp_1<eavlMinFunctor<float> >
        (array, minval, eavlMinFunctor<float>()), "find min");

    eavlExecutor::AddOperation(
        new eavlReduceOp_1<eavlMaxFunctor<float> >
        (array, maxval, eavlMaxFunctor<float>()), "find max");

    eavlExecutor::Go();

    int n = array->GetNumberOfTuples();

    float fmin = minval->GetComponentAsDouble(0,0);
    float fmax = maxval->GetComponentAsDouble(0,0);
    float fsize = fmax - fmin;

    eavlFloatArray *cutoffs = new eavlFloatArray("cutoffs", 1, nbins+1);
    for (int i = 0; i <= nbins; ++i)
        cutoffs->SetValue(i,  fmin + fsize * float(i) / float(nbins));

    eavlFloatArray *counts = new eavlFloatArray("counts", 1, nbins);
    eavlIntArray *inrange = new eavlIntArray("inrange", 1, n);
    eavlIntArray *tmpcount = new eavlIntArray("tmpcount", 1, 1);

    for (int bin = 0; bin<nbins; ++bin)
    {
        float lo = cutoffs->GetValue(bin);
        if (bin == 0)
            lo = -FLT_MAX;
        float hi = cutoffs->GetValue(bin+1);
        if (bin == nbins-1)
            hi = FLT_MAX;
        eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(array),
                                                 eavlOpArgs(inrange),
                                                 InRange(lo, hi)),
                                   "test if in range");
        eavlExecutor::AddOperation(new eavlReduceOp_1<eavlAddFunctor<int> >
               (inrange, tmpcount, eavlAddFunctor<int>()), "count in range");
        eavlExecutor::Go();
        counts->SetValue(bin, tmpcount->GetValue(0));
    }

    // create the output data set
    output->SetNumPoints(nbins+1);

    eavlRegularStructure reg;
    reg.SetCellDimension1D(nbins);

    output->SetLogicalStructure(new eavlLogicalStructureRegular(1, reg));

    eavlCoordinatesCartesian *coords =
        new eavlCoordinatesCartesian(output->GetLogicalStructure(),
                                     eavlCoordinatesCartesian::X);
    coords->SetAxis(0, new eavlCoordinateAxisField("cutoffs", 0));
    output->AddCoordinateSystem(coords);

    output->AddField(new eavlField(1,cutoffs,eavlField::ASSOC_POINTS));

    eavlCellSet *cellset = new eavlCellSetAllStructured("bins", reg);
    output->AddCellSet(cellset);

    output->AddField(new eavlField(0, counts, eavlField::ASSOC_CELL_SET, "bins"));
}
