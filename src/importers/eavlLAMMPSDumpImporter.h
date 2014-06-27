// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_LAMMPS_DUMP_IMPORTER_H
#define EAVL_LAMMPS_DUMP_IMPORTER_H

#include "STL.h"
#include "eavlDataSet.h"
#include "eavlImporter.h"


// ****************************************************************************
// Class:  eavlLAMMPSDumpImporter
//
// Purpose:
///   Import dump files output by LAMMPS
//
// Programmer:  Jeremy Meredith
// Creation:    December 18, 2013
//
// Modifications:
// ****************************************************************************
class eavlLAMMPSDumpImporter : public eavlImporter
{
  public:
    eavlLAMMPSDumpImporter(const string &fn)
    {
        currentTimestep = -1;
        metaDataRead = false;
        filename = fn;
        xIndex = yIndex = zIndex = speciesIndex = idIndex = -1;
        xMin = xMax = yMin = yMax = zMin = zMax = 0;

        ///\todo: not efficient to read it in the constructor
        ReadTimeStep(0);
    }
    virtual ~eavlLAMMPSDumpImporter()
    {
    }

    virtual vector<string> GetDiscreteDimNames()
    {
        return vector<string>(1, "time");
    }
    virtual vector<int>    GetDiscreteDimLengths()
    {
        ReadAllMetaData();
        return vector<int>(1, nTimeSteps);
    }
    vector<string>      GetCellSetList(const std::string &mesh)
    {
        return vector<string>(1,"cells");
    }
    vector<string> GetFieldList(const std::string &mesh)
    {
        return varNames;
    }
    int GetNumChunks(const std::string &mesh)
    {
        return 1;
    }
    virtual void SetDiscreteDim(int d, int i)
    {
        if (d != 0) // only one discrete dim: time
            throw; 

        if (currentTimestep != i)
            ReadTimeStep(i);
    }

    eavlDataSet   *GetMesh(const string &name, int chunk);
    eavlField     *GetField(const string &name, const string &mesh, int chunk);

  protected:
    ifstream                           in;
    std::vector<int>                   cycles;
    std::vector<istream::pos_type>     file_positions;
    std::string                        filename;
    bool                               metaDataRead;
    int                                nTimeSteps;
    int                                nVars;
    std::vector<int>                   nAtoms;
    double                             xMin, xMax;
    double                             yMin, yMax;
    double                             zMin, zMax;

    int                                currentTimestep;
    bool                               xScaled,yScaled,zScaled;
    int                                xIndex, yIndex, zIndex;
    int                                speciesIndex, idIndex;
    std::vector< std::vector<float> >  vars;
    std::vector<int>                   speciesVar, idVar;
    std::vector< std::string >         varNames;

    void OpenFileAtBeginning();
    void ReadTimeStep(int);
    void ReadAllMetaData();
};

#endif
