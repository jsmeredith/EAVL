// Copyright 2010-2012 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_CELL_SET_EXPLICIT_H
#define EAVL_CELL_SET_EXPLICIT_H

#include "eavlCellSet.h"
#include "eavlCellComponents.h"
#include "eavlArray.h"
#include "eavlException.h"
#include "eavlExplicitConnectivity.h"

// ****************************************************************************
// Class:  eavlCellSetExplicit
//
// Purpose:
///   A set of fully explicitly defined cells, like you encounter in
///   an unstructured grid.
//
// Programmer:  Jeremy Meredith, Dave Pugmire, Sean Ahern
// Creation:    February 15, 2011
//
// ****************************************************************************

class eavlCellSetExplicit : public eavlCellSet
{
  protected:
    ///\todo: shapetype is duplicated for each connectivity type
    /// (cellNode, cellEdge, etc.); can we unify them?
    eavlExplicitConnectivity cellNodeConnectivity;

    eavlExplicitConnectivity cellEdgeConnectivity;
    eavlExplicitConnectivity edgeNodeConnectivity;

    eavlExplicitConnectivity cellFaceConnectivity;
    eavlExplicitConnectivity faceNodeConnectivity;

    int numEdges;
    int numFaces;

    void BuildEdgeConnectivity();
    void BuildFaceConnectivity();
  public:
    eavlCellSetExplicit(const string &n, int d)
        : eavlCellSet(n,d),
          numEdges(-1),
          numFaces(-1)
    {
    }
    virtual int GetNumCells() { return cellNodeConnectivity.shapetype.size(); }
    virtual int GetNumEdges() { BuildEdgeConnectivity(); return numEdges; }
    virtual int GetNumFaces() { BuildFaceConnectivity(); return numFaces; }
    virtual void PrintSummary(ostream &out)
    {
        out << "    eavlCellSetExplicit:\n";
        //out << "        GetMemoryUsage returns = " << GetMemoryUsage() << endl;
        out << "        name = "<<name<<endl;
        out << "        dimensionality = "<<dimensionality<<endl;
        out << "        nCells = "<<GetNumCells()<<endl;
        out << "        cellNodeConnectivity =\n";
        cellNodeConnectivity.PrintSummary(out);
    }
    void SetCellNodeConnectivity(const eavlExplicitConnectivity &conn)
    {
        cellNodeConnectivity.Replace(conn);
        cellNodeConnectivity.CreateReverseIndex();
        numEdges = -1;
        numFaces = -1;
    }
    eavlExplicitConnectivity &GetConnectivity(eavlTopology topology)
    {
        ///\todo: this should return a *const* ref, except then we can't send
        /// it to the GPU because that required modified the conn device ptr.
        switch (topology)
        {
          case EAVL_NODES_OF_CELLS:
            return cellNodeConnectivity;

          case EAVL_NODES_OF_EDGES:
            BuildEdgeConnectivity();
            return edgeNodeConnectivity;

          case EAVL_NODES_OF_FACES:
            BuildFaceConnectivity();
            return faceNodeConnectivity;

          case EAVL_EDGES_OF_CELLS:
            BuildEdgeConnectivity();
            return cellEdgeConnectivity;

          case EAVL_FACES_OF_CELLS:
            BuildFaceConnectivity();
            return cellFaceConnectivity;
        }
        THROW(eavlException,"unexpected topology type in GetConnectivity");
    }
    virtual eavlCell GetCellNodes(int i)
    {
        eavlCell cell;
        int index = cellNodeConnectivity.mapCellToIndex[i];
        cell.type = (eavlCellShape)cellNodeConnectivity.shapetype[i];
        cell.numIndices = cellNodeConnectivity.connectivity[index];
        for (int n=0; n<cell.numIndices; n++)
            cell.indices[n] = cellNodeConnectivity.connectivity[index + 1 + n];
        return cell;
    }
    virtual eavlCell GetCellEdges(int i)
    {
        BuildEdgeConnectivity();
        eavlCell cell;
        int index = cellEdgeConnectivity.mapCellToIndex[i];
        cell.type = (eavlCellShape)cellEdgeConnectivity.shapetype[i];
        cell.numIndices = cellEdgeConnectivity.connectivity[index];
        for (int n=0; n<cell.numIndices; n++)
            cell.indices[n] = cellEdgeConnectivity.connectivity[index + 1 + n];
        return cell;
    }
    virtual eavlCell GetCellFaces(int i)
    {
        BuildFaceConnectivity();
        eavlCell cell;
        int index = cellFaceConnectivity.mapCellToIndex[i];
        cell.type = (eavlCellShape)cellFaceConnectivity.shapetype[i];
        cell.numIndices = cellFaceConnectivity.connectivity[index];
        for (int n=0; n<cell.numIndices; n++)
            cell.indices[n] = cellFaceConnectivity.connectivity[index + 1 + n];
        return cell;
    }
    virtual long long GetMemoryUsage()
    {
        long long mem = 0;
        mem += sizeof(vector<eavlCellShape>);
        mem += cellNodeConnectivity.shapetype.size() * sizeof(eavlCellShape);
        mem += sizeof(vector<int>);
        mem += cellNodeConnectivity.connectivity.size() * sizeof(int);
        mem += sizeof(vector<int>);
        mem += cellNodeConnectivity.mapCellToIndex.size() * sizeof(int);
        ///\todo: update this (e.g. with edge, face connectivity)
        return mem + eavlCellSet::GetMemoryUsage();
    }
};

#endif