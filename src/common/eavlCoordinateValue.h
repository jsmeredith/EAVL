// Copyright 2010-2012 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_COORDINATE_VALUE_H
#define EAVL_COORDINATE_VALUE_H

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

    void   SetValue(double inValue)        { value = inValue;   }
    double GetValue()                      { return value;      }
    
    virtual long long GetMemoryUsage()
    {
        return sizeof(double);
    }

  protected:
    double                  value;
};


#endif
