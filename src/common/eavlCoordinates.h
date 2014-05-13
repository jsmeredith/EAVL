// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_COORDINATES_H
#define EAVL_COORDINATES_H

#include "eavlField.h"
#include "eavlLogicalStructure.h"
#include "eavlException.h"


class eavlCoordinateAxis
{
  public:
    virtual ~eavlCoordinateAxis()
    {
    };
    
    virtual string className() const = 0;
    virtual eavlStream& serialize(eavlStream &s) const = 0;
    virtual eavlStream& deserialize(eavlStream &s) = 0;
    static eavlCoordinateAxis* CreateObjFromName(const string &nm);

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
    eavlCoordinateAxisField() : fieldName(""), component(0), fieldIndex(-1), fieldPointer(NULL) {}
    eavlCoordinateAxisField(const string &fn, int comp=0) :
        fieldName(fn), component(comp), fieldIndex(-1), fieldPointer(NULL) { }
    virtual ~eavlCoordinateAxisField()
    {
    }
    
    virtual string className() const {return "eavlCoordinateAxisField";}
    virtual eavlStream& serialize(eavlStream &s) const
    {
	s << className();
	s << fieldName << component << fieldIndex;
	return s;
    }
    virtual eavlStream& deserialize(eavlStream &s)
    {
	s >> fieldName >> component >> fieldIndex;
	return s;
    }

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
    eavlCoordinateAxisRegular() : logicaldim(-1), origin(0.0), delta(0.0) {}
    eavlCoordinateAxisRegular(int logicaldim_, double origin_, double delta_)
        : logicaldim(logicaldim_), origin(origin_), delta(delta_)
    {
    }
    virtual ~eavlCoordinateAxisRegular()
    {
    }
    
    virtual string className() const {return "eavlCoordinateAxisRegular";}
    virtual eavlStream& serialize(eavlStream &s) const
    {
	s << className();
	s << logicaldim << origin << delta;
	return s;
    }
    virtual eavlStream& deserialize(eavlStream &s)
    {
	s >> logicaldim >> origin >> delta;
	return s;
    }

    virtual void PrintSummary(ostream &out)
    {
        out << "          eavlCoordinateAxisRegular origin='"<<origin<<"' delta="<<delta<<endl;
    }
    virtual double GetValue(int pointIndex,
                            vector<int> &indexDivs,
                            vector<int> &indexMods,
                            vector<eavlField*>&)
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
    eavlCoordinates() {}
    eavlCoordinates(int ndims, eavlLogicalStructure *log) : axes(ndims)
    {
        axes.resize(ndims, NULL);
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
    virtual ~eavlCoordinates()
    {
        for (unsigned int i=0; i<axes.size(); ++i)
            delete axes[i];
    }
    
    virtual string className() const {return "eavlCoordinates";}
    virtual eavlStream& serialize(eavlStream &s) const
    {
	size_t sz = axes.size();
	s << sz;
	for (size_t i = 0; i < sz; i++)
	    axes[i]->serialize(s);
	s << indexMods << indexDivs;
	return s;
    }
    virtual eavlStream& deserialize(eavlStream &s)
    {
	size_t sz;
	s >> sz;
	axes.resize(sz);
	
	string nm;
	for (size_t i = 0; i < sz; i++)
	{
	    s >> nm;
	    axes[i] = eavlCoordinateAxis::CreateObjFromName(nm);
	    axes[i]->deserialize(s);
	}
	s >> indexMods >> indexDivs;
	return s;
    }
    static eavlCoordinates* CreateObjFromName(const string &nm);

    void SetAxis(int i, eavlCoordinateAxis *a)
    {
        if (axes[i])
            delete axes[i];
        axes[i] = a;
    }
    eavlCoordinateAxis *GetAxis(int i)
    {
        return axes[i];
    }
    virtual double GetRawPoint(int i, int c,
                               vector<eavlField*>&fd)
    {
        if (c >= (int)axes.size())
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
    bool IsCoordinateAxisField(const std::string &n)
    {
        for (unsigned int i=0; i<axes.size(); ++i)
        {
            eavlCoordinateAxisField *f = dynamic_cast<eavlCoordinateAxisField*>(axes[i]);
            if (f->GetFieldName() == n)
                return true;
        }
        return false;
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
    eavlCoordinatesCartesian() : eavlCoordinates() {axisMap[0]=axisMap[1]=axisMap[2]=-1;}
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
    virtual ~eavlCoordinatesCartesian()
    {
    }

    virtual string className() const {return "eavlCoordinatesCartesian";}
    virtual eavlStream& serialize(eavlStream &s) const
    {
	s << className();
	eavlCoordinates::serialize(s);
	s << axisTypes << axisMap[0] << axisMap[1] << axisMap[2];
	return s;
    }
    virtual eavlStream& deserialize(eavlStream &s)
    {
	eavlCoordinates::deserialize(s);
	s >> axisTypes >> axisMap[0] >> axisMap[1] >> axisMap[2];
	return s;
    }

    virtual double GetCartesianPoint(int i, int c,
                                     eavlLogicalStructure *,
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

#include <eavlMatrix4x4.h>
#include <eavlVector3.h>

class eavlCoordinatesCartesianWithTransform : public eavlCoordinatesCartesian
{
  protected:
    eavlMatrix4x4 transform;
  public:
    eavlCoordinatesCartesianWithTransform() {}
    eavlCoordinatesCartesianWithTransform(eavlLogicalStructure *log,
                             CartesianAxisType first)
        : eavlCoordinatesCartesian(log, first), transform()
    {
    }

    eavlCoordinatesCartesianWithTransform(eavlLogicalStructure *log,
                             CartesianAxisType first,
                             CartesianAxisType second)
        : eavlCoordinatesCartesian(log, first, second), transform()
    {
    }

    eavlCoordinatesCartesianWithTransform(eavlLogicalStructure *log,
                             CartesianAxisType first,
                             CartesianAxisType second,
                             CartesianAxisType third)
        : eavlCoordinatesCartesian(log, first, second, third), transform()
    {
    }
    virtual ~eavlCoordinatesCartesianWithTransform()
    {
    }

    virtual string className() const {return "eavlCoordinatesCartesianWithTransform";}
    virtual eavlStream& serialize(eavlStream &s) const
    {
	s << className();
	eavlCoordinatesCartesian::serialize(s);
	s.write((const char *)transform.m, sizeof(float)*4*4);
	return s;
    }
    virtual eavlStream& deserialize(eavlStream &s)
    {
	eavlCoordinatesCartesian::deserialize(s);
	s.read((char *)transform.m, sizeof(float)*4*4);
	return s;
    }

    void SetTransform(const eavlMatrix4x4 &m) { transform = m; }
    const eavlMatrix4x4 GetTransform() const { return transform; }

    // Well, my approach here is really lame. I'm going to end up transforming
    // each point 2-3 times because the points are requested later in a 
    // per-component fashion. I suppose that's because individual axes could
    // come from different eavl fields. I could cache the last point "i" but
    // that would make it non-threadsafe, though that may not matter.
    virtual double GetCartesianPoint(int i, int c,
                                     eavlLogicalStructure *,
                                     vector<eavlField*>&fd)
    {
        int axisIndex[3] = {-1,-1,-1};
        eavlVector3 coord(0., 0., 0.);
        for(int c2 = 0; c2 < GetDimension(); ++ c2)
        {
            axisIndex[c2] = axisMap[c2];
            if (axisIndex[c2] >= 0)
            {
        //cerr << "GetCartesianPoint, i="<<i<<" c="<<c<<" axisIndex="<<axisIndex<<endl;
                coord[c2] = GetRawPoint(i, axisIndex[c2], fd);
            }
        }

        eavlVector3 newcoord(transform * coord);
        return newcoord[c];
    }
};

#endif
