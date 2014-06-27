// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
// This file contains code from VisIt, (c) 2000-2014 LLNS.  See COPYRIGHT.txt.

#include "eavlLAMMPSDumpImporter.h"

#include <string.h>

// ****************************************************************************
//  Method:  eavlLAMMPSDumpImporter::OpenFileAtBeginning
//
//  Purpose:
//    Opens the file, or else seeks to the beginning.
//
//  Arguments:
//    none
//
//  Programmer:  Jeremy Meredith
//  Creation:    February  9, 2009
//
// ****************************************************************************
void
eavlLAMMPSDumpImporter::OpenFileAtBeginning()
{
    if (!in.is_open())
    {
        in.open(filename.c_str());
        if (!in)
        {
            THROW(eavlException, "Couldn't open file" + filename);
        }
    }
    else
    {
        in.clear();
        in.seekg(0, ios::beg);
    }
}


// ****************************************************************************
//  Method: GetMesh
//
//  Purpose:
//      Gets a mesh.
//
//  Programmer: Jeremy Meredith
//  Creation:   December 18, 2013
//
//  Modifications:
//
// ****************************************************************************

eavlDataSet *
eavlLAMMPSDumpImporter::GetMesh(const string &mesh, int)
{
    ReadTimeStep(currentTimestep);


    int n = nAtoms[currentTimestep];

    eavlDataSet *data = new eavlDataSet;
    data->SetNumPoints(n);

    // points
    eavlCoordinatesCartesian *coords = new eavlCoordinatesCartesian(NULL,
                                              eavlCoordinatesCartesian::X,
                                              eavlCoordinatesCartesian::Y,
                                              eavlCoordinatesCartesian::Z);

    data->AddCoordinateSystem(coords);
    coords->SetAxis(0,new eavlCoordinateAxisField("xcoord",0));
    coords->SetAxis(1,new eavlCoordinateAxisField("ycoord",0));
    coords->SetAxis(2,new eavlCoordinateAxisField("zcoord",0));

    eavlArray *axisValues[3] = {
        new eavlFloatArray("xcoord",1, n),
        new eavlFloatArray("ycoord",1, n),
        new eavlFloatArray("zcoord",1, n)
    };
    for (int i=0; i<n; i++)
    {
        double x = vars[xIndex][i];
        double y = vars[yIndex][i];
        double z = vars[zIndex][i];
        if (xScaled)
            x = xMin + (xMax-xMin) * x;
        if (yScaled)
            y = yMin + (yMax-yMin) * y;
        if (zScaled)
            z = zMin + (zMax-zMin) * z;
        axisValues[0]->SetComponentFromDouble(i, 0, x);
        axisValues[1]->SetComponentFromDouble(i, 0, y);
        axisValues[2]->SetComponentFromDouble(i, 0, z);
    }

    for (int d=0; d<3; d++)
    {
        eavlField *field = new eavlField(1, axisValues[d], eavlField::ASSOC_POINTS);
        data->AddField(field);
    }

    // atom cell set:
    // skip for now

    return data;
}


// ****************************************************************************
//  Method: GetField
//
//  Purpose:
//      Gets a scalar variable.
//
//  Programmer: Jeremy Meredith
//  Creation:   December 18, 2013
//
//  Modifications:
//
// ****************************************************************************

eavlField *
eavlLAMMPSDumpImporter::GetField(const string &varname, const string &mesh, int chunk)
{
    ReadTimeStep(currentTimestep);

    int n = nAtoms[currentTimestep];
    int varIndex = -1;
    for (int v=0; v<nVars; v++)
    {
        if (varNames[v] == varname)
        {
            varIndex = v;
            break;
        }
    }

    if (varIndex == -1)
    {
        THROW(eavlException, "Invalid variable: " + varname);
    }

    eavlArray *arr;
    if (string(varname) == "type")
    {
        eavlIntArray *iarr = new eavlIntArray(varname, 1, n);
        for (int i=0; i<n; ++i)
            iarr->SetValue(i, speciesVar[i]);
        arr = iarr;
    }
    else if (string(varname) == "id")
    {
        eavlIntArray *iarr = new eavlIntArray(varname, 1, n);
        for (int i=0; i<n; ++i)
            iarr->SetValue(i, idVar[i]);
        arr = iarr;
    }
    else
    {
        eavlFloatArray *farr = new eavlFloatArray(varname, 1, n);
        for (int i=0; i<n; ++i)
            farr->SetValue(i, vars[varIndex][i]);
        arr = farr;
    }

    return new eavlField(0, arr, eavlField::ASSOC_POINTS);
}


// ****************************************************************************
//  Method:  eavlLAMMPSDumpImporter::ReadTimeStep
//
//  Purpose:
//    Read only the atoms for the given time step.
//
//  Arguments:
//    timestep   the time state for which to read the atoms
//
//  Programmer:  Jeremy Meredith
//  Creation:    February  9, 2009
//
//  Modifications:
//
// ****************************************************************************
void
eavlLAMMPSDumpImporter::ReadTimeStep(int timestep)
{
    ReadAllMetaData();

    // don't read this time step if it's already in memory
    if (currentTimestep == timestep)
        return;
    currentTimestep = timestep;

    OpenFileAtBeginning();
    in.seekg(file_positions[timestep]);

    speciesVar.resize(nAtoms[timestep]);
    idVar.resize(nAtoms[timestep]);
    for (int v=0; v<int(vars.size()); v++)
    {
        // id and species are ints; don't bother with the float arrays for them
        if (v == idIndex || v == speciesIndex)
            continue;
        vars[v].resize(nAtoms[timestep]);
    }

    std::vector<double> tmpVars(nVars);
    int tmpID, tmpSpecies;

    char buff[1000];
    // read all the atoms
    for (int a=0; a<nAtoms[timestep]; a++)
    {
        in.getline(buff,1000);
        istringstream sin(buff);
        for (int v=0; v<nVars; v++)
        {
            if (v==speciesIndex)
                sin >> tmpSpecies;
            else if (v==idIndex)
                sin >> tmpID;
            else
                sin >> tmpVars[v];
        }

        int index = a;  // no longer tmpID (tmpID-1 actually); don't re-sort
        for (int v=0; v<nVars; v++)
        {
            if (v == idIndex || v == speciesIndex)
                continue;
            vars[v][index] = tmpVars[v];
        }
        speciesVar[index] = tmpSpecies - 1;
        idVar[index] = tmpID;
    }
}


// ****************************************************************************
//  Method:  eavlLAMMPSDumpImporter::ReadMetaData
//
//  Purpose:
//    The metadata we need to read here is (a) count the number of
//    time steps, and (b) count how many entries are in the atoms
//    so we know how many variables to report.
//
//  Arguments:
//    none
//
//  Programmer:  Jeremy Meredith
//  Creation:    February  9, 2009
//
//  Modifications:
//
// ****************************************************************************
void
eavlLAMMPSDumpImporter::ReadAllMetaData()
{
    if (metaDataRead)
        return;

    OpenFileAtBeginning();

    char buff[1000];

    nTimeSteps = 0;
    nVars = -1;

    while (in)
    {
        in.getline(buff,1000);
        if (strncmp(buff, "ITEM:", 5) != 0)
            continue;

        string item(&buff[6]);
        if (item == "TIMESTEP")
        {
            nTimeSteps++;
            in.getline(buff,1000);
            cycles.push_back(strtol(buff, NULL, 10));
        }
        else if (item.substr(0,19) == "BOX BOUNDS xy xz yz")
        {
            float xy, xz, yz;
            in >> xMin >> xMax >> xy;
            in >> yMin >> yMax >> xz;
            in >> zMin >> zMax >> yz;
            in.getline(buff, 1000); // get rest of Z line
        }
        else if (item.substr(0,10) == "BOX BOUNDS")
        {
            in >> xMin >> xMax;
            in >> yMin >> yMax;
            in >> zMin >> zMax;
            in.getline(buff, 1000); // get rest of Z line
        }
        else if (item == "NUMBER OF ATOMS")
        {
            in.getline(buff,1000);
            int n = strtol(buff, NULL, 10);
            nAtoms.push_back(n);
        }
        else if (item.substr(0,5) == "ATOMS")
        {
            istream::pos_type current_pos = in.tellg();
            file_positions.push_back(current_pos);
            if (nVars == -1)
            {
                istringstream sin(&buff[11]);
                string varName;
                xScaled = yScaled = zScaled = false;
                while (sin >> varName)
                {
                    if (varName == "id")
                        idIndex = (int)varNames.size();
                    else if (varName == "type")
                        speciesIndex = (int)varNames.size();
                    else if (varName == "x" || varName == "xs" || 
                               varName == "xu" || varName == "xsu" )
                        xIndex = (int)varNames.size();
                    else if (varName == "y" || varName == "ys" ||
                               varName == "yu" || varName == "ysu" )
                        yIndex = (int)varNames.size();
                    else if (varName == "z" || varName == "zs" ||
                               varName == "zu" || varName == "zsu" )
                        zIndex = (int)varNames.size();

                    if (varName == "xs" || "xsu")
                        xScaled = true;
                    if (varName == "ys" || "ysu")
                        yScaled = true;
                    if (varName == "zs" || "zsu")
                        zScaled = true;

                    varNames.push_back(varName);

                }
                nVars = (int)varNames.size();
                if (nVars == 0)
                {
                    // OLD FORMAT: Assume "id type x y z"
                    varNames.push_back("id");
                    varNames.push_back("type");
                    varNames.push_back("x");
                    varNames.push_back("y");
                    varNames.push_back("z");
                    idIndex = 0;
                    speciesIndex = 1;
                    xIndex = 2; xScaled = false;
                    yIndex = 3; yScaled = false;
                    zIndex = 4; zScaled = false;
                    nVars = (int)varNames.size();
                }
                vars.resize(nVars);
            }
        }
    }

    if (xIndex<0 || yIndex<0 || zIndex<0 || idIndex<0 || speciesIndex<0)
    {
        THROW(eavlException, "Bad file " + filename +
              ": Didn't get indices for all necessary vars");
    }

    // don't read the meta data more than once
    metaDataRead = true;
}
