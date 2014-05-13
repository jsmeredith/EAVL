// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_CELL_SET_ALL_POINTS_H
#define EAVL_CELL_SET_ALL_POINTS_H

// ****************************************************************************
// Class:  eavlCellSetAllPoints
//
// Purpose:
///   The set of cells which is all points of the dataset.
///   \todo: this has not been used yet.
//
// Programmer:  Jeremy Meredith, Dave Pugmire, Sean Ahern
// Creation:    February 15, 2011
//
// ****************************************************************************

class eavlCellSetAllPoints : public eavlCellSet
{
  public:
    eavlCellSetAllPoints(const string &n) : eavlCellSet(n, 0) { }
    
    virtual string className() const {return "eavlCellSetAllPoints";}
    virtual eavlStream& serialize(eavlStream &s) const
    {
	s << className();
	eavlCellSet::serialize(s);
	return s;
    }
    virtual eavlStream& deserialize(eavlStream &s)
    {
	eavlCellSet::deserialize(s);
	return s;
    }
    
    virtual void PrintSummary(ostream &out) const
    {
        out << "    eavlCellSetAllPoints:\n";
        out << "        name = "<<name<<endl;
        out << "        dimensionality = "<<dimensionality<<endl;
        out << "        nCells = "<<GetNumCells()<<endl;
    }
    virtual eavlCell GetCellNodes(int i)
    {
        eavlCell cell;
        cell.type = SHAPETYPE_POINT;
        cell.numIndices = 1;
        cell.indices[0] = i;
        return cell;
    }
    virtual int GetNumCells()
    {
        ///\todo: unimplemented
        throw;
    }
    virtual long long GetMemoryUsage()
    {
        return eavlCellSet::GetMemoryUsage();
    }
    
};

#endif
