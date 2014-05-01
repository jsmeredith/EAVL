// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavlDataSet.h"
#include "eavlArray.h"
#include "eavlCellSetExplicit.h"
#include "eavlCellSetAllStructured.h"
#include "eavlCoordinates.h"
#include "eavlLogicalStructureRegular.h"
#include "eavlVTKExporter.h"

//
// Create a Rectilinear Grid in the XY plane
// 
eavlDataSet *GenerateRectXY(int ni, int nj)
{
    eavlDataSet *data = new eavlDataSet();

    // set the number of points
    int npts = ni * nj;
    data->SetNumPoints(npts);

    eavlRegularStructure reg;
    reg.SetNodeDimension2D(ni, nj);

    // set the logical structure
    eavlLogicalStructure *log = new eavlLogicalStructureRegular(reg.dimension,
                                                                reg);
    data->SetLogicalStructure(log);

    // create the coordinate axes
    eavlFloatArray *x = new eavlFloatArray("x", 1, ni);
    for (int i=0; i<ni; ++i)
        x->SetValue(i, 100 + 10*i);

    eavlFloatArray *y = new eavlFloatArray("y", 1, nj);
    for (int j=0; j<nj; ++j)
        y->SetValue(j, 200 + 15*j);

    // add the coordinate axis arrays as linear fields on logical dims
    data->AddField(new eavlField(1, x, eavlField::ASSOC_LOGICALDIM, 0));
    data->AddField(new eavlField(1, y, eavlField::ASSOC_LOGICALDIM, 1));

    // set the coordinates
    eavlCoordinates *coords = new eavlCoordinatesCartesian(log,
                                            eavlCoordinatesCartesian::X,
                                            eavlCoordinatesCartesian::Y);
    coords->SetAxis(0, new eavlCoordinateAxisField("x"));
    coords->SetAxis(1, new eavlCoordinateAxisField("y"));
    data->AddCoordinateSystem(coords);

    // create a cell set implicitly covering the entire regular structure
    eavlCellSet *cells = new eavlCellSetAllStructured("cells", reg);
    data->AddCellSet(cells);

    return data;
}

//
// Create a Rectilinear Grid in the XZ plane
// 
eavlDataSet *GenerateRectXZ(int ni, int nj)
{
    eavlDataSet *data = new eavlDataSet();

    // set the number of points
    int npts = ni * nj;
    data->SetNumPoints(npts);

    eavlRegularStructure reg;
    reg.SetNodeDimension2D(ni, nj);

    // set the logical structure
    eavlLogicalStructure *log = new eavlLogicalStructureRegular(reg.dimension,
                                                                reg);
    data->SetLogicalStructure(log);

    // create the coordinate axes
    eavlFloatArray *x = new eavlFloatArray("x", 1, ni);
    for (int i=0; i<ni; ++i)
        x->SetValue(i, 100 + 10*i);

    eavlFloatArray *z = new eavlFloatArray("z", 1, nj);
    for (int j=0; j<nj; ++j)
        z->SetValue(j, 200 + 15*j);

    // add the coordinate axis arrays as linear fields on logical dims
    data->AddField(new eavlField(1, x, eavlField::ASSOC_LOGICALDIM, 0));
    data->AddField(new eavlField(1, z, eavlField::ASSOC_LOGICALDIM, 1));

    // set the coordinates
    eavlCoordinates *coords = new eavlCoordinatesCartesian(log,
                                            eavlCoordinatesCartesian::X,
                                            eavlCoordinatesCartesian::Z);
    coords->SetAxis(0, new eavlCoordinateAxisField("x"));
    coords->SetAxis(1, new eavlCoordinateAxisField("z"));
    data->AddCoordinateSystem(coords);

    // create a cell set implicitly covering the entire regular structure
    eavlCellSet *cells = new eavlCellSetAllStructured("cells", reg);
    data->AddCellSet(cells);

    return data;
}

//
// Create a Topologically 2D Curvilinear Grid in 3D space
// 
eavlDataSet *GenerateCurv2Din3D(int ni, int nj)
{
    eavlDataSet *data = new eavlDataSet();

    // set the number of points
    int npts = ni * nj;
    data->SetNumPoints(npts);

    eavlRegularStructure reg;
    reg.SetNodeDimension2D(ni, nj);

    // set the logical structure
    eavlLogicalStructure *log = new eavlLogicalStructureRegular(reg.dimension,
                                                                reg);
    data->SetLogicalStructure(log);

    // create the coordinate axes
    eavlFloatArray *x = new eavlFloatArray("x", 1, npts);
    eavlFloatArray *y = new eavlFloatArray("y", 1, npts);
    eavlFloatArray *z = new eavlFloatArray("z", 1, npts);
    for (int p=0; p<npts; ++p)
    {
        int i = p % ni;
        int j = p / ni;
        x->SetValue(p, 100 + 10*i + (j%3)*3);
        y->SetValue(p, 100 + i-.1*j*j);
        z->SetValue(p, 200 + 15*j + 2*sin((double)i));
    }

    // add the coordinate axis arrays as linear fields on logical dims
    data->AddField(new eavlField(1, x, eavlField::ASSOC_POINTS));
    data->AddField(new eavlField(1, y, eavlField::ASSOC_POINTS));
    data->AddField(new eavlField(1, z, eavlField::ASSOC_POINTS));

    // set the coordinates
    eavlCoordinates *coords = new eavlCoordinatesCartesian(log,
                                            eavlCoordinatesCartesian::X,
                                            eavlCoordinatesCartesian::Y,
                                            eavlCoordinatesCartesian::Z);
    coords->SetAxis(0, new eavlCoordinateAxisField("x"));
    coords->SetAxis(1, new eavlCoordinateAxisField("y"));
    coords->SetAxis(2, new eavlCoordinateAxisField("z"));
    data->AddCoordinateSystem(coords);

    // create a cell set implicitly covering the entire regular structure
    eavlCellSet *cells = new eavlCellSetAllStructured("cells", reg);
    data->AddCellSet(cells);

    return data;
}

//
// Create a grid in 3D space that's rectilinear in X/Y and explicit in Z
// Essentially, this is a hybrid rectilinear/curvilinear grid
//
eavlDataSet *GenerateRectXY_ElevToZ(int ni, int nj)
{
    eavlDataSet *data = new eavlDataSet();

    // set the number of points
    int npts = ni * nj;
    data->SetNumPoints(npts);

    eavlRegularStructure reg;
    reg.SetNodeDimension2D(ni, nj);

    // set the logical structure
    eavlLogicalStructure *log = new eavlLogicalStructureRegular(reg.dimension,
                                                                reg);
    data->SetLogicalStructure(log);

    // create the coordinate axes
    eavlFloatArray *x = new eavlFloatArray("x", 1, ni);
    for (int i=0; i<ni; ++i)
        x->SetValue(i, 100 + 10*i);

    eavlFloatArray *y = new eavlFloatArray("y", 1, nj);
    for (int j=0; j<nj; ++j)
        y->SetValue(j, 200 + 15*j);

    eavlFloatArray *z = new eavlFloatArray("z", 1, npts);
    for (int p=0; p<npts; ++p)
        z->SetValue(p, 4 * (p%3) + 5 * sqrt((double)p));

    // add the coordinate axis arrays as linear fields on logical dims
    data->AddField(new eavlField(1, x, eavlField::ASSOC_LOGICALDIM, 0));
    data->AddField(new eavlField(1, y, eavlField::ASSOC_LOGICALDIM, 1));
    data->AddField(new eavlField(1, z, eavlField::ASSOC_POINTS));

    // set the coordinates
    eavlCoordinates *coords = new eavlCoordinatesCartesian(log,
                                            eavlCoordinatesCartesian::X,
                                            eavlCoordinatesCartesian::Y,
                                            eavlCoordinatesCartesian::Z);
    coords->SetAxis(0, new eavlCoordinateAxisField("x"));
    coords->SetAxis(1, new eavlCoordinateAxisField("y"));
    coords->SetAxis(2, new eavlCoordinateAxisField("z"));
    data->AddCoordinateSystem(coords);

    // create a cell set implicitly covering the entire regular structure
    eavlCellSet *cells = new eavlCellSetAllStructured("cells", reg);
    data->AddCellSet(cells);

    return data;
}

//
// Create an explict (unstructured) grid in 3D space
//
eavlDataSet *GenerateExplicit3DPyramidGrid(int ni, int nj)
{
    eavlDataSet *data = new eavlDataSet();

    // set the number of points
    int npts = 1 + ni * nj;
    data->SetNumPoints(npts);

    // no logical structure
    eavlLogicalStructure *log = NULL;

    // create the coordinate axes
    eavlFloatArray *x = new eavlFloatArray("x", 1, npts);
    eavlFloatArray *y = new eavlFloatArray("y", 1, npts);
    eavlFloatArray *z = new eavlFloatArray("z", 1, npts);
    for (int p=0; p<npts; ++p)
    {
        if (p == 0)
        {
            x->SetValue(0, 0.0);
            y->SetValue(0, 0.0);
            z->SetValue(0, 0.0);
        }
        else
        {
            int i = (p-1) % ni;
            int j = (p-1) / ni;
            x->SetValue(p, 100 * cos(i / 10.) * cos((j-2)/10.));
            y->SetValue(p, 100 * sin(i / 10.) * cos((j-2)/10.));
            z->SetValue(p, 100 * sin((j-2)/10.));
        }
    }

    // add the coordinate axis arrays as linear fields on points
    data->AddField(new eavlField(1, x, eavlField::ASSOC_POINTS));
    data->AddField(new eavlField(1, y, eavlField::ASSOC_POINTS));
    data->AddField(new eavlField(1, z, eavlField::ASSOC_POINTS));

    // set the coordinates
    eavlCoordinates *coords = new eavlCoordinatesCartesian(log,
                                            eavlCoordinatesCartesian::X,
                                            eavlCoordinatesCartesian::Y,
                                            eavlCoordinatesCartesian::Z);
    coords->SetAxis(0, new eavlCoordinateAxisField("x"));
    coords->SetAxis(1, new eavlCoordinateAxisField("y"));
    coords->SetAxis(2, new eavlCoordinateAxisField("z"));
    data->AddCoordinateSystem(coords);

    // create a topologically 3D cell set with pyramids
    eavlCellSetExplicit *cells = new eavlCellSetExplicit("cells", 3);
    eavlExplicitConnectivity conn;
    eavlFloatArray *indexarray = new eavlFloatArray("index", 1);
    for (int j=0; j < (nj-1); ++j)
    {
        for (int i=0; i < (ni-1); ++i)
        {
            int cellpts[5] = { 1 + ni * (j+0) + i+0,
                               1 + ni * (j+0) + i+1,
                               1 + ni * (j+1) + i+1,
                               1 + ni * (j+1) + i+0,
                               0 };
            conn.AddElement(EAVL_PYRAMID, 5, cellpts);
            indexarray->AddValue(j * (ni-1) + i);
        }
    }
    cells->SetCellNodeConnectivity(conn);
    data->AddCellSet(cells);

    data->AddField(new eavlField(0, indexarray, eavlField::ASSOC_CELL_SET, "cells"));

    return data;
}

//
// Create a grid that's logically 2D but has an explicit cell set as well
//
eavlDataSet *GenerateExplicitOnLogical(int ni, int nj)
{
    eavlDataSet *data = new eavlDataSet();

    // set the number of points
    int npts = ni * nj;
    data->SetNumPoints(npts);

    eavlRegularStructure reg;
    reg.SetNodeDimension2D(ni, nj);

    // set the logical structure
    eavlLogicalStructure *log = new eavlLogicalStructureRegular(reg.dimension,
                                                                reg);
    data->SetLogicalStructure(log);

    // create the coordinate axes
    eavlFloatArray *x = new eavlFloatArray("x", 1, ni);
    for (int i=0; i<ni; ++i)
        x->SetValue(i, 100 + 10*i);

    eavlFloatArray *y = new eavlFloatArray("y", 1, nj);
    for (int j=0; j<nj; ++j)
        y->SetValue(j, 200 + 15*j);

    // add the coordinate axis arrays as linear fields on logical dims
    data->AddField(new eavlField(1, x, eavlField::ASSOC_LOGICALDIM, 0));
    data->AddField(new eavlField(1, y, eavlField::ASSOC_LOGICALDIM, 1));

    // set the coordinates
    eavlCoordinates *coords = new eavlCoordinatesCartesian(log,
                                            eavlCoordinatesCartesian::X,
                                            eavlCoordinatesCartesian::Y);
    coords->SetAxis(0, new eavlCoordinateAxisField("x"));
    coords->SetAxis(1, new eavlCoordinateAxisField("y"));
    data->AddCoordinateSystem(coords);

    // create a topologically 2D cell set explicitly containing some cells
    // create a cell field of something interesting
    eavlCellSetExplicit *cells = new eavlCellSetExplicit("cells", 2);
    eavlExplicitConnectivity conn;
    eavlFloatArray *cellarray = new eavlFloatArray("cellarray", 1);
    for (int j=0; j<nj-1; ++j)
    {
        for (int i=0; i<ni-1; ++i)
        {
            bool included = sqrt((double)(i*i + j*j)) >= 8;
            if (!included)
                continue;
            int cellpts[4] = { reg.CalculateNodeIndex2D(i+0, j+0),
                               reg.CalculateNodeIndex2D(i+1, j+0),
                               reg.CalculateNodeIndex2D(i+0, j+1),
                               reg.CalculateNodeIndex2D(i+1, j+1) };
                            
            conn.AddElement(EAVL_PIXEL, 4, cellpts);
            cellarray->AddValue(reg.CalculateCellIndex2D(i,j));
        }
    }
    cells->SetCellNodeConnectivity(conn);
    data->AddCellSet(cells);

    data->AddField(new eavlField(0, cellarray, eavlField::ASSOC_CELL_SET, "cells"));

    return data;
}

//
// Create a molecule requiring two cell sets
//
eavlDataSet *GenerateMoleculeTwoCellSets()
{
    eavlDataSet *data = new eavlDataSet();

    // set the number of points
    int npts = 3;
    data->SetNumPoints(npts);

    // no logical structure
    eavlLogicalStructure *log = NULL;

    // create the 3D coordinate axes as 3-component array
    eavlFloatArray *coordarray = new eavlFloatArray("coords", 3, npts);
    float h0xyz[] = {-1, 0,  -.3};
    float o1xyz[] = {-.2, 1, -.1};
    float h2xyz[] = {+1, 0.3, .3};
    coordarray->SetTuple(0, h0xyz);
    coordarray->SetTuple(1, o1xyz);
    coordarray->SetTuple(2, h2xyz);

    // add the coordinate axis array as a linear fields on points
    data->AddField(new eavlField(1, coordarray, eavlField::ASSOC_POINTS));

    // set the coordinates
    eavlCoordinates *coords = new eavlCoordinatesCartesian(log,
                                            eavlCoordinatesCartesian::X,
                                            eavlCoordinatesCartesian::Y,
                                            eavlCoordinatesCartesian::Z);
    coords->SetAxis(0, new eavlCoordinateAxisField("coords", 0));
    coords->SetAxis(1, new eavlCoordinateAxisField("coords", 1));
    coords->SetAxis(2, new eavlCoordinateAxisField("coords", 2));
    data->AddCoordinateSystem(coords);

    // create a topologically 0D cell set for the atoms
    eavlExplicitConnectivity aconn;
    eavlIntArray *speciesarray = new eavlIntArray("species", 1);
    int index;
    int species;
    // add hydrogen, then oxygen, then hydrogen
    index=0; species=1;
    aconn.AddElement(EAVL_POINT, 1, &index);
    speciesarray->AddValue(species);
    index=1; species=8;
    aconn.AddElement(EAVL_POINT, 1, &index);
    speciesarray->AddValue(species);
    index=2; species=1;
    aconn.AddElement(EAVL_POINT, 1, &index);
    speciesarray->AddValue(species);

    eavlCellSetExplicit *atoms = new eavlCellSetExplicit("atoms", 0);
    atoms->SetCellNodeConnectivity(aconn);
    data->AddCellSet(atoms);
    data->AddField(new eavlField(0, speciesarray, eavlField::ASSOC_CELL_SET, "atoms"));

    // create a topologically 1D cell set for the bonds
    eavlExplicitConnectivity bconn;
    eavlFloatArray *strengtharray = new eavlFloatArray("strength", 1);
    int endpts[2];
    float strength;
    endpts[0] = 0; endpts[1] = 1;  strength = 1.0;
    bconn.AddElement(EAVL_BEAM, 2, endpts);
    strengtharray->AddValue(strength);
    endpts[0] = 1; endpts[1] = 2;  strength = 1.0;
    bconn.AddElement(EAVL_BEAM, 2, endpts);
    strengtharray->AddValue(strength);


    eavlCellSetExplicit *bonds = new eavlCellSetExplicit("bonds", 1);
    bonds->SetCellNodeConnectivity(bconn);
    data->AddCellSet(bonds);
    data->AddField(new eavlField(0, strengtharray, eavlField::ASSOC_CELL_SET, "bonds"));

    return data;
}

//
// Print a summary of the new data set and write to a file
//
void test(const char *fn, eavlDataSet *ds)
{
    cerr << "------------ " << fn << " -----------\n";
    ds->PrintSummary(cout);

    ofstream out(fn);

    eavlVTKExporter exporter(ds, 0);
    exporter.Export(out);
    out.close();

}

//
// Test all the cases
//
int main(int argc, char *argv[])
{
    try
    {
        test("rectxy.vtk",       GenerateRectXY(10, 12));
        test("rectxz.vtk",       GenerateRectXZ(10, 12));
        test("curv2d3d.vtk",     GenerateCurv2Din3D(10, 12));
        test("rectxyelev.vtk",   GenerateRectXY_ElevToZ(10, 12));
        test("unstruc3d.vtk",    GenerateExplicit3DPyramidGrid(10, 12));
        test("explicitrect.vtk", GenerateExplicitOnLogical(10, 12));
        test("molecule.vtk",     GenerateMoleculeTwoCellSets());
    }
    catch (const eavlException &e)
    {
        cerr << e.GetErrorText() << endl;
        cerr << "\nUsage: "<<argv[0]<<" <infile>\n";
        return 1;
    }

    return 0;
}
