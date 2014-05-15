// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_DATA_SET_H
#define EAVL_DATA_SET_H

#include "eavlCoordinates.h"
#include "eavlLogicalStructure.h"
#include "eavlCoordinateValue.h"
#include "eavlCellSet.h"
#include "eavlField.h"
#include "eavlPoint3.h"
#include "eavlException.h"
#include "eavlIndexable.h"
#include "eavlSerialize.h"

// ****************************************************************************
// Class:  eavlDataSet
//
// Purpose:
///   A single, identifiable chunk of problem data.  This might be the
///   equivalent of a "domain", for example, where it has a discrete coordinate
///   value for time, and the points cover a region of X/Y/Z Cartesian space.
///   \todo: How do we handle things like a linear transform for the coords?
///   \todo: discrete coords are very different from how the continuous
///       coords are now stored; we could e.g. just make this a single
///       eavlCoordinates, but I'm not sure what I think of that.
///       Maybe better: these are now simply single-valued FIELDS,
///       with an association to the whole-mesh? 
//
// Programmer:  Jeremy Meredith, Dave Pugmire, Sean Ahern
// Creation:    February 15, 2011
//
// ****************************************************************************
class eavlDataSet
{
  protected:
    int                          npoints;
    vector<eavlCoordinateValue>  discreteCoordinates;
    vector<eavlField*>           fields;
    vector<eavlCellSet*>         cellsets;
    vector<eavlCoordinates*>     coordinateSystems;
    eavlLogicalStructure        *logicalStructure;

  public:
    eavlDataSet()
    {
        npoints = 0;
        logicalStructure = NULL;
    }
    ~eavlDataSet()
    {
        Clear();
    }
    virtual string className() const {return "eavlDataSet";}
    virtual eavlStream& serialize(eavlStream &s) const;
    virtual eavlStream& deserialize(eavlStream &s);

    eavlIndexable<eavlArray> GetIndexableAxis(int i, eavlCoordinates *coordsys = NULL)
    {
        if (!coordsys)
            coordsys = GetCoordinateSystem(0);
        if (coordsys->GetDimension() <= i)
            THROW(eavlException, "Tried to get more axes than spatial dimensions");

        eavlCoordinateAxisField *axis = dynamic_cast<eavlCoordinateAxisField*>(coordsys->GetAxis(i));
        if (!axis)
            THROW(eavlException,"Expected only field-based coordinate axes");

        eavlField *field = GetField(axis->GetFieldName());
        eavlArray *array = field->GetArray();
        if (!array)
            THROW(eavlException, "Problem obtaining coordinate array");

        eavlLogicalStructureRegular *logReg = dynamic_cast<eavlLogicalStructureRegular*>(logicalStructure);
        eavlIndexable<eavlArray> indexable(array, axis->GetComponent());
        if (field->GetAssociation() == eavlField::ASSOC_WHOLEMESH)
            indexable.indexer.mul = 0;
        if (logReg)
        {
            eavlRegularStructure &reg = logReg->GetRegularStructure();
            if (field->GetAssociation() == eavlField::ASSOC_LOGICALDIM)
                indexable = eavlIndexable<eavlArray>(array, axis->GetComponent(), reg, field->GetAssocLogicalDim());
        }
        return indexable;
    }
    void Clear()
    {
        discreteCoordinates.clear();
        if (logicalStructure)
            delete logicalStructure;
        logicalStructure = NULL;
        for (unsigned int i=0; i<cellsets.size(); ++i)
        {
            if (cellsets[i])
                delete cellsets[i];
        }
        cellsets.clear();
        for (unsigned int i=0; i<fields.size(); ++i)
        {
            if (fields[i])
                delete fields[i];
        }
        fields.clear();
        for (unsigned int i=0; i<coordinateSystems.size(); ++i)
        {
            if (coordinateSystems[i])
                delete coordinateSystems[i];
        }
        coordinateSystems.clear();
        npoints = 0;
    }
    eavlDataSet *CreateShallowCopy()
    {
        eavlDataSet *data = new eavlDataSet;
        data->npoints             = npoints;
        data->discreteCoordinates = discreteCoordinates;
        data->fields              = fields;
        data->cellsets            = cellsets;
        data->coordinateSystems   = coordinateSystems;
        data->logicalStructure    = logicalStructure;
        return data;
    }
    
    int GetNumPoints()
    {
        return npoints;
    }

    void SetNumPoints(int n)
    {
        npoints = n;
        for (unsigned int i=0; i<cellsets.size(); i++)
        {
            cellsets[i]->SetDSNumPoints(npoints);
        }
    }

    double GetPoint(int i, int c, int whichCoordSystem=0)
    {
        assert(whichCoordSystem >= 0 && whichCoordSystem <= (int)coordinateSystems.size());
        /// \todo: this assumes you have at least one coordinate system
        /// and that you want to use the first one; bad assumptions.
        /// \todo: I don't like how we pass in the field data.
        return coordinateSystems[whichCoordSystem]->
            GetCartesianPoint(i,c,logicalStructure,fields);
    }

    long long GetMemoryUsage()
    {
        long long mem = 0;
        mem += sizeof(int); //npoints;

        mem += sizeof(vector<eavlCoordinateValue>);
        for (size_t i=0; i<discreteCoordinates.size(); i++)
            mem += discreteCoordinates[i].GetMemoryUsage();

        mem += sizeof(eavlLogicalStructure*);

        mem += sizeof(vector<eavlCoordinates*>);
        mem += coordinateSystems.size() * sizeof(eavlCoordinates*);
        for (size_t i=0; i<coordinateSystems.size(); i++)
            mem += coordinateSystems[i]->GetMemoryUsage();

        mem += sizeof(vector<eavlCellSet*>);
        mem += cellsets.size() * sizeof(eavlCellSet*);
        for (size_t i=0; i<cellsets.size(); i++)
        {
            //cerr << "   cellsets["<<i<<"] memory usage = "<<cellsets[i]->GetMemoryUsage()<<endl;
            mem += cellsets[i]->GetMemoryUsage();
        }

        mem += sizeof(vector<eavlField*>);
        mem += fields.size() * sizeof(eavlField*);
        for (size_t i=0; i<fields.size(); i++)
        {
            //cerr << "   fields["<<i<<"] memory usage = "<<fields[i]->GetMemoryUsage()<<endl;
            mem += fields[i]->GetMemoryUsage();
        }

        //cerr << "  total memory usage = "<<mem<<endl;
        return mem;
    }

    eavlLogicalStructure *GetLogicalStructure()
    {
        return logicalStructure;
    }

    void SetLogicalStructure(eavlLogicalStructure *log)
    {
        logicalStructure = log;
    }

    int GetNumCoordinateSystems()
    {
        return coordinateSystems.size();
    }

    eavlCoordinates* GetCoordinateSystem(int index)
    {
        return coordinateSystems[index];
    }

    void AddCoordinateSystem(eavlCoordinates *cs)
    {
        coordinateSystems.push_back(cs);
    }

    void SetCoordinateSystem(int index, eavlCoordinates *cs)
    {
        coordinateSystems[index] = cs;
    }

    virtual int GetNumCellSets()
    {
        return cellsets.size();
    }

    eavlCellSet *GetCellSet(const string &name)
    {
        int index = GetCellSetIndex(name);
        if (index < 0)
        {
            THROW(eavlException,"Couldn't find cell set");
        }
        else
            return cellsets[index];
    }
    
    eavlCellSet *GetCellSet(int index)
    {
        return cellsets[index];
    }
    
    int GetCellSetIndex(const string &name)
    {
        int n = cellsets.size();
        if (n <= 0)
            THROW(eavlException,"No cell sets to return");

        ///\todo: decide if we want to support a default cell set name
        /*
        // default = empty string
        if (name == "")
            return 0;
        */

        for (int i=0; i<n; i++)
            if (cellsets[i]->GetName() == name)
                return i;

        return -1;
    }
    
    void AddCellSet(eavlCellSet *c)
    {
        cellsets.push_back(c);
        c->SetDSNumPoints(npoints);
    }

    virtual int GetNumFields()
    {
        return fields.size();
    }

    int GetFieldIndex(const string &name)
    {
        int n = fields.size();
        if (n <= 0)
            THROW(eavlException,"No fields to return");

        for (int i=0; i<n; i++)
            if (fields[i]->GetArray()->GetName() == name)
                return i;

        return -1;
    }
    
    eavlField *GetField(const string &name)
    {
        int index = GetFieldIndex(name);
        if (index < 0)
        {
            THROW(eavlException,"Couldn't find field");
        }
        else
            return fields[index];
    }
    
    eavlField *GetField(unsigned int idx)
    {
        assert(idx < fields.size());
        if (idx < fields.size())
            return fields[idx];
        THROW(eavlException,"Couldn't find field");
    }

    void AddField(eavlField *f)
    {
        fields.push_back(f);
    }

    void PrintSummary(ostream &out)
    {
        out << "eavlDataSet:\n";
	int origPrecision = out.precision();
	out << setprecision(4);
        //out << "   GetMemoryUsage() reports: "<<GetMemoryUsage()<<endl;
        out << "   npoints = "<<npoints << endl;
        if (logicalStructure)
            logicalStructure->PrintSummary(out);
        out << "   coordinateSystems["<<coordinateSystems.size()<<"]:\n";
        for (unsigned int i=0; i<coordinateSystems.size(); i++)
        {
            coordinateSystems[i]->PrintSummary(out);
        }
        out << "  discreteCoordinates["<<discreteCoordinates.size()<<"]:\n";
        for (unsigned int i=0; i<discreteCoordinates.size(); i++)
        {
            out << "    discrete axis #" << i
                << " = " <<  discreteCoordinates[i].GetValue()
                << endl;
        }
        out << "  cellsets["<<cellsets.size()<<"]:\n";
        for (unsigned int i=0; i<cellsets.size(); i++)
        {
            cellsets[i]->PrintSummary(out);
        }
        out << "  fields["<<fields.size()<<"]:\n";
        eavlField::iterator it;
        for (it = fields.begin(); it != fields.end(); it++)
        {
            (*it)->PrintSummary(out);
        }
	
	out << setprecision(origPrecision);
    }
};

//API

// ****************************************************************************
// Function:  AddStructuredMesh
//
// Purpose:
///  Add a structured mesh, and optional cellsets to a data set.
//
// Programmer:  Dave Pugmire
// Creation:    July 22, 2011
//
// ****************************************************************************

int
AddRectilinearMesh(eavlDataSet *data,
                   const vector<vector<double> > &coordinates,
                   const vector<string> &coordinateNames,
                   bool addCellSet, string cellSetName="");

int
AddCurvilinearMesh(eavlDataSet *data,
                   int dims[3],
                   const vector<vector<double> > &coordinates,
                   const vector<string> &coordinateNames,
                   bool addCellSet, string cellSetName="");

int
AddCurvilinearMesh_SepCoords(eavlDataSet *data,
                   int dims[3],
                   const vector<vector<double> > &coordinates,
                   const vector<string> &coordinateNames,
                   bool addCellSet, string cellSetName="");


#endif
