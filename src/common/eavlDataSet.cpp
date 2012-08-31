// Copyright 2010-2012 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavlDataSet.h"
#include "eavlCellSetAllStructured.h"
#include "eavlException.h"

// ****************************************************************************
// Function:  AddRectilinearMesh
//
// Purpose:
///  Add a rectilinear mesh, and optional cellsets to a data set.
//
// Programmer:  Dave Pugmire
// Creation:    July 22, 2011
//
// Modifications:
// ****************************************************************************

int
AddRectilinearMesh(eavlDataSet *data,
                   const vector< vector<double> > &coordinates,
                   const vector<string> &coordinateNames,
                   bool addCellSet,
                   string cellSetName)
{
    if (data->coordinateSystems.size() != 0)
        THROW(eavlException,"Error: multiple meshes not supported!");
    if (coordinates.size() != coordinateNames.size() || coordinates.size() == 0)
        THROW(eavlException,"Error: coordinates and coordinatesNames must be the same size, and > 0");
        
    int meshIndex = -1;
    int dimension = 0;
    int dims[3];
    int ctr = 0;
    eavlCoordinatesCartesian::CartesianAxisType axes[3];
    data->npoints = 1;
    for (unsigned int d = 0; d < coordinates.size(); d++)
    {
        int n = coordinates[d].size();
        if (n > 1)
        {
            data->npoints *= n;
            dims[dimension] = n;
            dimension++;
        }
        axes[ctr++] = (d==0 ? eavlCoordinatesCartesian::X :
                       (d==1 ? eavlCoordinatesCartesian::Y : 
                        eavlCoordinatesCartesian::Z));
    }


    eavlRegularStructure reg;
    if (dimension == 1)
        reg.SetNodeDimension1D(dims[0]);
    else if (dimension == 2)
        reg.SetNodeDimension2D(dims[0],dims[1]);
    else if (dimension == 3)
        reg.SetNodeDimension3D(dims[0],dims[1],dims[2]);
    else
        THROW(eavlException,"unxpected number of dimensions");

    eavlLogicalStructureRegular *log =
        new eavlLogicalStructureRegular(dimension, reg);

    eavlCoordinatesCartesian *coords = NULL;
    if (coordinates.size() == 1)
        coords = new eavlCoordinatesCartesian(log, axes[0]);
    else if (coordinates.size() == 2)
        coords = new eavlCoordinatesCartesian(log, axes[0], axes[1]);
    else if (coordinates.size() == 3)
        coords = new eavlCoordinatesCartesian(log, axes[0], axes[1], axes[2]);
    else
    {
        THROW(eavlException,"unxpected number of dimensions");
    }

    int ldim = 0;
    
    for (unsigned int d = 0; d < coordinates.size(); d++)
    {
        eavlFloatArray *c = new eavlFloatArray(coordinateNames[d], 1);
        
        c->SetNumberOfTuples(coordinates[d].size());

        vector<double>::const_iterator iter;
        int i = 0;
        for (iter = coordinates[d].begin(); iter != coordinates[d].end(); ++iter, ++i)
            c->SetComponentFromDouble(i, 0, *iter);
        
        eavlField *cField = NULL;
        if (coordinates[d].size() > 1)
        {
            cField = new eavlField(1, c, eavlField::ASSOC_LOGICALDIM, ldim);
            ldim++;
        }
        else
            cField = new eavlField(1, c, eavlField::ASSOC_WHOLEMESH);
        
        data->fields.push_back(cField);

        coords->SetAxis(d, new eavlCoordinateAxisField(coordinateNames[d]));

    }

    

    data->coordinateSystems.push_back(coords);
    meshIndex = data->coordinateSystems.size()-1;
    data->logicalStructure = log;

    if (addCellSet)
    {
        int nCells = 1;
        for (int i = 0; i < dimension; i++)
        {
            int n_i = coordinates[i].size()-1;
            if (n_i > 1)
                nCells *= n_i;
        }
        
        eavlCellSetAllStructured *cellset =
            new eavlCellSetAllStructured(cellSetName, reg);
        data->cellsets.push_back(cellset);
    }

    return meshIndex;
}

// ****************************************************************************
// Function:  AddCurvilinearMesh
//
// Purpose:
///  Add a curvilinear mesh, and optional cellsets to a data set.
///  This version uses a single 3-component array for the coordinates.
//
// Programmer:  Dave Pugmire
// Creation:    July 22, 2011
//
// Modifications:
// ****************************************************************************

int
AddCurvilinearMesh(eavlDataSet *data,
                   int dims[3],
                   const vector<vector<double> > &coordinates,
                   const vector<string> &coordinateNames,
                   bool addCellSet,
                   string cellSetName)
{
    ///\todo: this doesn't properly strip dims[?]==1
    if (data->coordinateSystems.size() != 0)
        THROW(eavlException,"Error: multiple meshes not supported!");
    if (coordinates.size() != coordinateNames.size() || coordinates.size() == 0)
        THROW(eavlException,"Error: coordinates and coordinatesNames must be the same size, and > 0");

    int meshIndex = -1;
    int dimension = coordinateNames.size();

    data->npoints = 1;
    eavlCoordinatesCartesian::CartesianAxisType axes[3];
    int ctr = 0;
    for (unsigned int d = 0; d < coordinates.size(); d++)
    {
        data->npoints *= dims[d];
        axes[ctr++] = (d==0 ? eavlCoordinatesCartesian::X :
                       (d==1 ? eavlCoordinatesCartesian::Y : 
                        eavlCoordinatesCartesian::Z));
    }

    eavlRegularStructure reg;
    if (dimension == 1)
        reg.SetNodeDimension1D(dims[0]);
    else if (dimension == 2)
        reg.SetNodeDimension2D(dims[0],dims[1]);
    else if (dimension == 3)
        reg.SetNodeDimension3D(dims[0],dims[1],dims[2]);
    else
        THROW(eavlException,"unxpected number of dimensions");

    eavlLogicalStructureRegular *log =
        new eavlLogicalStructureRegular(dimension, reg);

    eavlCoordinatesCartesian *coords = NULL;
    if (dimension == 1)
        coords = new eavlCoordinatesCartesian(log, axes[0]);
    else if (dimension == 2)
        coords = new eavlCoordinatesCartesian(log, axes[0], axes[1]);
    else if (dimension == 3)
        coords = new eavlCoordinatesCartesian(log, axes[0], axes[1], axes[2]);
    else
    {
        THROW(eavlException,"unxpected number of dimensions");
    }

    for (int d = 0; d < dimension; d++)
    {
        coords->SetAxis(d, new eavlCoordinateAxisField("coords", d));
    }
        
    eavlFloatArray *c = new eavlFloatArray("coords", dimension);
    c->SetNumberOfTuples(data->npoints);

    for (int d = 0; d < dimension; d++)
    {
        for (unsigned int i=0; i<coordinates[d].size(); i++)
            c->SetComponentFromDouble(i,d, coordinates[d][i]);
    }        
     
    eavlField *cField = new eavlField(1, c, eavlField::ASSOC_POINTS);
    data->fields.push_back(cField);


    data->coordinateSystems.push_back(coords);
    meshIndex = data->coordinateSystems.size()-1;
    data->logicalStructure = log;

    if (addCellSet)
    {
        int nCells = 1;
        for (int i = 0; i < dimension; i++)
        {
            int n_i = dims[i]-1;
            if (n_i > 1)
                nCells *= n_i;
        }
        
        eavlCellSetAllStructured *cellset =
            new eavlCellSetAllStructured(cellSetName, reg);
        data->cellsets.push_back(cellset);
    }

    return meshIndex;
}

// ****************************************************************************
// Function:  AddCurvilinearMesh_SepCoords
//
// Purpose:
///  Add a curvilinear mesh, and optional cellsets to a data set.
///  This version uses a 3 single-component arrays for the coordinates.
//
// Programmer:  Jeremy Meredith
// Creation:    February 3, 2012
//
// Modifications:
// ****************************************************************************

int
AddCurvilinearMesh_SepCoords(eavlDataSet *data,
                   int dims[3],
                   const vector<vector<double> > &coordinates,
                   const vector<string> &coordinateNames,
                   bool addCellSet,
                   string cellSetName)
{
    ///\todo: this doesn't properly strip dims[?]==1
    if (data->coordinateSystems.size() != 0)
        THROW(eavlException,"Error: multiple meshes not supported!");
    if (coordinates.size() != coordinateNames.size() || coordinates.size() == 0)
        THROW(eavlException,"Error: coordinates and coordinatesNames must be the same size, and > 0");

    int meshIndex = -1;
    int dimension = coordinateNames.size();

    data->npoints = 1;
    eavlCoordinatesCartesian::CartesianAxisType axes[3];
    int ctr = 0;
    for (int d = 0; d < dimension; d++)
    {
        data->npoints *= dims[d];
        axes[ctr++] = (d==0 ? eavlCoordinatesCartesian::X :
                       (d==1 ? eavlCoordinatesCartesian::Y : 
                        eavlCoordinatesCartesian::Z));
    }

    eavlRegularStructure reg;
    if (dimension == 1)
        reg.SetNodeDimension1D(dims[0]);
    else if (dimension == 2)
        reg.SetNodeDimension2D(dims[0],dims[1]);
    else if (dimension == 3)
        reg.SetNodeDimension3D(dims[0],dims[1],dims[2]);
    else
        THROW(eavlException,"unxpected number of dimensions");

    eavlLogicalStructureRegular *log =
        new eavlLogicalStructureRegular(dimension, reg);

    eavlCoordinatesCartesian *coords = NULL;
    if (dimension == 1)
        coords = new eavlCoordinatesCartesian(log, axes[0]);
    else if (dimension == 2)
        coords = new eavlCoordinatesCartesian(log, axes[0], axes[1]);
    else if (dimension == 3)
        coords = new eavlCoordinatesCartesian(log, axes[0], axes[1], axes[2]);
    else
    {
        THROW(eavlException,"unxpected number of dimensions");
    }


    for (int d = 0; d < dimension; d++)
    {
        eavlFloatArray *c = new eavlFloatArray(coordinateNames[d], 1);
        c->SetNumberOfTuples(coordinates[d].size());

        for (unsigned int i=0; i<coordinates[d].size(); i++)
            c->SetComponentFromDouble(i,0, coordinates[d][i]);
     
        eavlField *cField = new eavlField(1, c, eavlField::ASSOC_POINTS);
        data->fields.push_back(cField);

        coords->SetAxis(d, new eavlCoordinateAxisField(coordinateNames[d]));
    }

    data->coordinateSystems.push_back(coords);
    meshIndex = data->coordinateSystems.size()-1;
    data->logicalStructure = log;

    if (addCellSet)
    {
        int nCells = 1;
        for (int i = 0; i < dimension; i++)
        {
            int n_i = dims[i]-1;
            if (n_i > 1)
                nCells *= n_i;
        }
        
        eavlCellSetAllStructured *cellset =
            new eavlCellSetAllStructured(cellSetName, reg);
        data->cellsets.push_back(cellset);
    }

    return meshIndex;
}
