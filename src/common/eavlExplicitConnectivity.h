// Copyright 2010-2012 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_EXPLICIT_CONNECTIVITY_H
#define EAVL_EXPLICIT_CONNECTIVITY_H

#include "eavlArray.h"
#include "eavlFlatArray.h"
#include "eavlCell.h"
#include "eavlTopology.h"
#include "eavlUtility.h"

/*
// Exploring ways to improve the cpu-gpu device copies.
// The safest way may be to make a device-specific version
// that the original host version can generate.  This has
// an added benefit that ray pointers use fewer bytes, and
// we're limited on bytes for parameter args to cuda kernels
// (256 bytes for 1.0-1.3, 4096 bytes for 2.0 and up.  3.0/3.5
// may be even higher, I haven't checked yet.)  For example:
struct eavlExplicitConnectivity_device
{
    int *shapetype;
    int *connectivity;
    int *mapCellToIndex;
};
*/

// ****************************************************************************
// Class:  eavlExplicitConnectivity
//
// Purpose:
///   An explicit list of topology containment for grids.
///   For example, this might be a list of nodes for each cell,
///   or a list of cells for each face.
//
// Programmer:  Jeremy Meredith
// Creation:    July 25, 2012
//
// Modifications:
// ****************************************************************************
struct eavlExplicitConnectivity
{
    eavlFlatArray<int> shapetype;
    eavlFlatArray<int> connectivity;
    eavlFlatArray<int> mapCellToIndex;

    eavlExplicitConnectivity()
    {
    }
    eavlExplicitConnectivity(const eavlExplicitConnectivity &e)
        : shapetype(e.shapetype),
          connectivity(e.connectivity),
          mapCellToIndex(e.mapCellToIndex)
    {
    }
    /*
    eavlExplicitConnectivity_device GetDeviceVersion()
    {
        shapetype.NeedOnDevice();
        connectivity.NeedOnDevice();
        mapCellToIndex.NeedOnDevice();
        eavlExplicitConnectivity_device d;
        d.shapetype = shapetype.device;
        d.connectivity = connectivity.device;
        d.mapCellToIndex = mapCellToIndex.device;
        return d;
    }*/
    /*
    eavlExplicitConnectivity(const eavlExplicitConnectivity &e)
    {
        cerr << "eavlExplicitConnectivity: copy constructor (old style)\n";
        shapetype = e.shapetype;
        connectivity = e.connectivity;
        mapCellToIndex = e.mapCellToIndex;
        cerr << "e.shapetype.copied="<<e.shapetype.copied<<"\n";
        cerr << "shapetype.copied="<<shapetype.copied<<"\n";
    }
    */
    
    int GetNumElements() const { return shapetype.size(); }
    void AddElement(eavlCellShape shape, int npts, int *conn)
    {
        connectivity.push_back(npts);
        for (int i=0; i<npts; i++)
            connectivity.push_back(conn[i]);
        shapetype.push_back(int(shape));
    }
    /// \todo: surface normal only needs 3 nodes; can we improve its
    /// performance by only having it return three values in that case?
    EAVL_HOSTDEVICE int GetElementComponents(int index, int &npts, int *pts) const
    {
        int ci = mapCellToIndex[index];
        npts = connectivity[ci];
        for (int i=0; i<npts; ++i)
            pts[i] = connectivity[ci + 1 + i];
        return shapetype[index];
    }
    EAVL_HOSTONLY void CreateReverseIndex()
    {
        int nCells = shapetype.size();
        mapCellToIndex.resize(nCells);
        int index = 0;
        for (int e=0; e<nCells; e++)
        {
            mapCellToIndex[e] = index;
            int npts = connectivity[index];
            index += (npts + 1); // "+1" for the npts value
        }
    }
    EAVL_HOSTONLY void Replace(const eavlExplicitConnectivity &e)
    {
        // since we track whether or not a flat array was copied --
        // because managing the CUDA device memory is tricky --
        // we need a way to replace this connectivity with a new 
        // one, without using the assignment op or copy constructor.
        int ns = e.shapetype.size();
        shapetype.resize(ns);
        for (int i=0; i<ns; i++)
            shapetype[i] = e.shapetype[i];

        int nc = e.connectivity.size();
        connectivity.resize(nc);
        for (int i=0; i<nc; i++)
            connectivity[i] = e.connectivity[i];

        mapCellToIndex.clear();
    }
    EAVL_HOSTONLY void PrintSummary(ostream &out)
    {
        out << "        shapetype["<<shapetype.size()<<"] = ";
        PrintVectorSummary(out, shapetype);
        out << endl;
        out << "        connectivity["<<connectivity.size()<<"] = ";
        PrintVectorSummary(out, connectivity);
        out << endl;
        out << "        mapCellToIndex["<<mapCellToIndex.size()<<"] = ";
        PrintVectorSummary(out, mapCellToIndex);
        out << endl;        
    }
};

#endif
