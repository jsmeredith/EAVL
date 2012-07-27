// Copyright 2010-2012 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_COORDINATES_H
#define EAVL_COORDINATES_H

#include "eavlField.h"
#include "eavlLogicalStructure.h"
#include "eavlException.h"


class eavlCoordinateAxis
{
  public:
    virtual double GetValue(int pointIndex,
                            vector<int> &indexDivs,
                            vector<int> &indexMods,
                            vector<eavlField*>&fd) = 0;
    virtual void PrintSummary(ostream &out) = 0;
    virtual long long GetMemoryUsage()
    {
        return 0;
    }
};

class eavlCoordinateAxisField : public eavlCoordinateAxis
{
  protected:
    string fieldName;
    int    component;

    /// for cache verification
    int                 fieldIndex;
    /// cached, and for cache verification
    eavlField *fieldPointer;
  public:
    eavlCoordinateAxisField(const string &fn, int comp=0) :
        fieldName(fn), component(comp), fieldIndex(-1), fieldPointer(NULL) { }
    virtual void PrintSummary(ostream &out)
    {
        out << "          eavlCoordinateAxisField='"<<fieldName<<"',component #"<<component<<endl;
    }
    ///\todo: it's a weird mismatch here; can't we just pass down the
    /// REAL index into this array, instead of both and basing it off the assoc?
    virtual double GetValue(int pointIndex,
                            vector<int> &indexDivs,
                            vector<int> &indexMods,
                            vector<eavlField*>&fd)
    {
        // if the index is out of bounds or the pointer to the field changes
        // then we need to find the right field by name again
        if (fieldIndex < 0           ||
            fieldIndex >= (int)fd.size()  ||
            fd[fieldIndex] != fieldPointer)
        {
            fieldIndex = -1;
            for (unsigned int f=0; f<fd.size(); f++)
            {
                if (fd[f]->GetArray()->GetName() == fieldName)
                {
                    fieldIndex = f;
                    fieldPointer = fd[f];
                }
            }
            if (fieldIndex < 0)
                THROW(eavlException,"Can't find field matchng coordinate field name");
        }
        switch (fieldPointer->GetAssociation())
        {
          case eavlField::ASSOC_WHOLEMESH:
            return fieldPointer->GetArray()->GetComponentAsDouble(0, component);
          case eavlField::ASSOC_POINTS:
            return fieldPointer->GetArray()->GetComponentAsDouble(pointIndex, component);
          case eavlField::ASSOC_LOGICALDIM:
            {
            int div = indexDivs[fieldPointer->GetAssocLogicalDim()];
            int mod = indexMods[fieldPointer->GetAssocLogicalDim()];
            int logicalIndex = (pointIndex / div) % mod;
            return fieldPointer->GetArray()->GetComponentAsDouble(logicalIndex, component);
            }
          default:
            THROW(eavlException,"unexpected association for eavlCoordinateAxisField");
        }
        
    }
    string GetFieldName()
    {
        return fieldName;
    }
    int GetComponent()
    {
        return component;
    }
    virtual long long GetMemoryUsage()
    {
        long long mem = 0;
        mem += sizeof(fieldName);
        mem += sizeof(component);
        return mem;
    }
};

class eavlCoordinateAxisRegular : public eavlCoordinateAxis
{
  protected:
    int    logicaldim;
    double origin, delta;        
  public:
    ///\todo: obviously it's useless to pass field data down
    /// for a regular array
    eavlCoordinateAxisRegular(int logicaldim_, double origin_, double delta_)
        : logicaldim(logicaldim_), origin(origin_), delta(delta_)
    {
    }
    virtual void PrintSummary(ostream &out)
    {
        out << "          eavlCoordinateAxisRegular origin='"<<origin<<"' delta="<<delta<<endl;
    }
    virtual double GetValue(int pointIndex,
                            vector<int> &indexDivs,
                            vector<int> &indexMods,
                            vector<eavlField*>&fd)
    {
        int div = indexDivs[logicaldim];
        int mod = indexMods[logicaldim];
        int logicalIndex = (pointIndex / div) % mod;
        return origin + logicalIndex*delta;
    }
    virtual long long GetMemoryUsage()
    {
        return 2 * sizeof(double);
    }
};

// ****************************************************************************
// Class:  eavlCoordinates
//
// Purpose:
///   It's a coordinate system!
///   \todo: I think we can rename to eavlCoordinateSystem at this point.
//
// Programmer:  Jeremy Meredith, Dave Pugmire, Sean Ahern
// Creation:    February 17, 2012
//
// ****************************************************************************
class eavlCoordinates
{
  protected:
    vector<eavlCoordinateAxis*> axes;
    vector<int>                 indexMods; /// note: of length nLogDims, not nSpatDims
    vector<int>                 indexDivs; /// note: of length nLogDims, not nSpatDims
  public:
    eavlCoordinates(int ndims, eavlLogicalStructure *log) : axes(ndims)
    {
        eavlLogicalStructureRegular *l =
            dynamic_cast<eavlLogicalStructureRegular*>(log);
        if (l)
        {
            indexMods.resize(l->GetDimension(), 1);
            indexDivs.resize(l->GetDimension(), 1);
            for (int d=0; d<l->GetDimension(); d++)
            {
                indexMods[d] = l->GetRegularStructure().CalculateNodeIndexModForDimension(d);
                indexDivs[d] = l->GetRegularStructure().CalculateNodeIndexDivForDimension(d);
            }
        }
    }
    void SetAxis(int i, eavlCoordinateAxis *a)
    {
        axes[i] = a;
    }
    eavlCoordinateAxis *GetAxis(int i)
    {
        return axes[i];
    }
    virtual double GetRawPoint(int i, int c,
                               vector<eavlField*>&fd)
    {
        if (c >= axes.size())
            THROW(eavlException,"asked for a component we didn't have in our coordinates");
        return axes[c]->GetValue(i,
                                 indexDivs,
                                 indexMods,
                                 fd);
    }
    virtual double GetCartesianPoint(int i,int c,
                                     eavlLogicalStructure *log,
                                     vector<eavlField*>&fd)=0;
    virtual void PrintSummary(ostream &out)
    {
        out << "   eavlCoordinates, ndims="<<axes.size()<<": " << endl;
        for (size_t d=0; d<indexMods.size(); d++)
            out << "       logdim "<<d<<": div="<<indexDivs[d]<<",mod="<<indexMods[d]<<"\n";
        for (size_t a=0; a<axes.size(); a++)
        {
            out << "       axis "<<a<<":\n";
            axes[a]->PrintSummary(out);
        }
    }
    virtual int GetDimension()
    {
        return axes.size();
    }
    virtual long long GetMemoryUsage()
    {
        long long mem = 0;
        mem += sizeof(vector<int>) + indexMods.size() * sizeof(int);
        mem += sizeof(vector<int>) + indexDivs.size() * sizeof(int);
        mem += sizeof(vector<eavlCoordinateAxis*>);
        for (size_t i=0; i<axes.size(); i++)
            mem += axes[i]->GetMemoryUsage();
        return mem;
    }
};


class eavlCoordinatesCartesian : public eavlCoordinates
{
  public:
    enum CartesianAxisType { X=0, Y=1, Z=2 };
  protected:
    vector<CartesianAxisType> axisTypes;
    int axisMap[3];
  public:
    eavlCoordinatesCartesian(eavlLogicalStructure *log,
                             CartesianAxisType first)
        : eavlCoordinates(1, log)
    {
        axisTypes.push_back(first);
        axisMap[0] = axisMap[1] = axisMap[2] = -1;
        axisMap[int(first)] = 0;
    }
    eavlCoordinatesCartesian(eavlLogicalStructure *log,
                             CartesianAxisType first,
                             CartesianAxisType second)
        : eavlCoordinates(2, log)
    {
        axisTypes.push_back(first);
        axisTypes.push_back(second);
        axisMap[0] = axisMap[1] = axisMap[2] = -1;
        axisMap[int(first)]  = 0;
        axisMap[int(second)] = 1;
    }
    eavlCoordinatesCartesian(eavlLogicalStructure *log,
                             CartesianAxisType first,
                             CartesianAxisType second,
                             CartesianAxisType third)
        : eavlCoordinates(3, log)
    {
        axisTypes.push_back(first);
        axisTypes.push_back(second);
        axisTypes.push_back(third);
        axisMap[0] = axisMap[1] = axisMap[2] = -1;
        axisMap[int(first)]  = 0;
        axisMap[int(second)] = 1;
        axisMap[int(third)]  = 2;
    }
    virtual double GetCartesianPoint(int i, int c,
                                     eavlLogicalStructure *log,
                                     vector<eavlField*>&fd)
    {
        int axisIndex = axisMap[c];
        if (axisIndex < 0)
            return 0; // default value
        //cerr << "GetCartesianPoint, i="<<i<<" c="<<c<<" axisIndex="<<axisIndex<<endl;
        return GetRawPoint(i, axisIndex, fd);
    }
    virtual void PrintSummary(ostream &out)
    {
        out << "   eavlCoordinatesCartesian, ndims="<<axes.size()<<": " << endl;
        for (size_t d=0; d<indexMods.size(); d++)
            out << "       logdim "<<d<<": div="<<indexDivs[d]<<",mod="<<indexMods[d]<<"\n";
        for (size_t a=0; a<axes.size(); a++)
        {
            out << "       axis "<<a
                << " (type="<<char((int('X')+int(axisTypes[a])))<<"):\n";
            axes[a]->PrintSummary(out);
        }
    }
};

#endif
