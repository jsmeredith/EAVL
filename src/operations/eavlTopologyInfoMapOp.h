// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_TOPOLOGY_INFO_MAP_OP_H
#define EAVL_TOPOLOGY_INFO_MAP_OP_H

// ****************************************************************************
// Class:  eavlTopologyInfoMapOp
//
// Purpose:
///   Map from input to output arrays on the same topology type.
///
///   Much like a standard map, in that it does an element-wise map
///   between arrays of the same length, but the fields are known to
///   be on some sort of topology type -- typically a cell.  Or, you
///   could instead think of it like an eavlTopologyMap, but both the
///   inputs and outputs are on the same topology type.  (Or like an
///   eavlCombinedTopologyMap, but without a source topology.)
///
///   Essentially, this just adds a "shapetype" to the functor call of
///   a standard map operation.  For example, a cell-to-cell map would
///   be a simple map, but with the shape type (e.g. EAVL_HEX or
///   EAVL_TET) passed along with every functor call.
//
// Programmer:  Jeremy Meredith
// Creation:    August  1, 2013
//
// Modifications:
// ****************************************************************************

#endif
