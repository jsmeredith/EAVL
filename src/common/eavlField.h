// Copyright 2010-2012 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_FIELD_H
#define EAVL_FIELD_H

#include "eavlCellSet.h"
#include "eavlException.h"

// ****************************************************************************
// Class:  eavlField
//
// Purpose:
///   An array associated with a mesh.  It may be associated with the
///   cells, points, or the whole mesh.
//
// Programmer:  Jeremy Meredith, Dave Pugmire, Sean Ahern
// Creation:    March  1, 2011
//
// ****************************************************************************

///\todo: add more about higher-order and where e.g. shared face vals are stored
class eavlField
{
  public:
    typedef vector<eavlField *>::const_iterator iterator;

    enum Association
    {
        ASSOC_WHOLEMESH,
        ASSOC_POINTS,
        ASSOC_LOGICALDIM,
        ASSOC_CELL_SET
    };

  protected:
    int          order; ///< 0=(piecewise) constant, 1=linear, 2=quadratic
    Association  association;

    ///\todo: don't like these floating here like a union
    ///       if assoc_cell_set and assoc_logicaldim are both the same
    ///       type, they could just be a single value (assoc_index), right?
    ///\todo: assoc_cell_set could be a string, right?
    int          assoc_cell_set;  ///< only populate if assoc is cells
    ///\todo: don't like these floating here like a union
    ///\todo: other question: do we even really need this?  it seems like
    ///       most often the reference would be from coordsys -> fielddata,
    ///       not from fielddata -> back to coord system
    ///\todo: we're assuming a logical dimension must be a NODE array.
    ///       couldn't it be a CELL logical array?  
    int          assoc_logicaldim; ///< only populate if assoc is logical dim
    
    eavlArray   *array;

  public:
    eavlField(int order_,
              eavlArray *a,
              Association assoc,
              int assoc_value = -1)
        : order(order_),
          association(assoc),
          assoc_cell_set(assoc_value),
          assoc_logicaldim(assoc_value),
          array(a)
    {
        if (assoc == ASSOC_CELL_SET && assoc_value < 0)
            THROW(eavlException,"Need a nonnegative cell set index for cell set association");
        if (assoc == ASSOC_LOGICALDIM && assoc_value < 0)
            THROW(eavlException,"Need a nonnegative dim index for logical dim association");
    }
    eavlField(eavlField *f,
              eavlArray *a)
        : order(f->order),
          association(f->association),
          assoc_cell_set(f->assoc_cell_set),
          assoc_logicaldim(f->assoc_logicaldim),
          array(a)
    {
    }
    ~eavlField()
    {
        delete array;
    }
    Association  GetAssociation()    {return association;}
    eavlArray   *GetArray()          {return array;}
    int          GetOrder()          {return order;}
    int          GetAssocCellSet()   {return assoc_cell_set;}
    int          GetAssocLogicalDim(){return assoc_logicaldim;}

    virtual void PrintSummary(ostream &out)
    {
        out << "      array name = "
            <<array->GetName()
            <<endl;
        //out << "      GetMemoryUsage returns = " << GetMemoryUsage() << endl;
        out << "      order = " << order << endl;
        out << "      association = "
            << (association==ASSOC_WHOLEMESH?"WHOLEMESH":
                (association==ASSOC_POINTS?"POINTS":
                 (association==ASSOC_LOGICALDIM?"LOGICALDIM":
                  (association==ASSOC_CELL_SET?"CELL_SET":"<unknown>"))))
            << endl;
        if (association == ASSOC_CELL_SET)
            out << "      assoc_cell_set (index) = " << assoc_cell_set << endl;
        if (association == ASSOC_LOGICALDIM)
            out << "      assoc_logicaldim = " << assoc_logicaldim << endl;
        out << "      array = ";
        array->PrintSummary(out);
        out << endl;
    }
    virtual long long GetMemoryUsage()
    {
        long long mem = 0;
        mem += sizeof(bool);
        mem += sizeof(Association);
        mem += sizeof(int);
        mem += sizeof(int);
        mem += array->GetMemoryUsage();
        return mem;
    }
};

#endif
