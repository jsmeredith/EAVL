// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_COORDINATE_VALUE_H
#define EAVL_COORDINATE_VALUE_H

#include "eavlSerialize.h"

// ****************************************************************************
// Class:  eavlCoordinateValue
//
// Purpose:
///   Implements a coordinate value.
//
// Programmer:  Jeremy Meredith, Dave Pugmire, Sean Ahern
// Creation:    February 15, 2011
//
// ****************************************************************************

class eavlCoordinateValue
{
  public:
    eavlCoordinateValue () {}
    eavlCoordinateValue (double inValue)
    {
        value      = inValue;
    }
    virtual string className() const {return "eavlCoordinateValue";}
    virtual eavlStream& serialize(eavlStream &s) const
    {
	s<<className()<<value;
	return s;
    }
    virtual eavlStream& deserialize(eavlStream &s)
    {
	string nm;
	s >> nm;
	s >> value;
	return s;
    }
    
    void   SetValue(double inValue)        { value = inValue;   }
    double GetValue() const                { return value;      }
    
    virtual long long GetMemoryUsage()
    {
        return sizeof(double);
    }

  protected:
    double                  value;
};


#endif
