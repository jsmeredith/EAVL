// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_REGULAR_STRUCTURE_H
#define EAVL_REGULAR_STRUCTURE_H

#include "eavlCell.h"
#include "eavlTopology.h"
#include "eavlUtility.h"
#include "eavlSerialize.h"

// ****************************************************************************
// Class:  eavlRegularStructure
//
// Purpose:
///   Defines connectivity topologies for a regular grid.
//
// Programmer:  Jeremy Meredith
// Creation:    July 25, 2012
//
// Modifications:
//   Jeremy Meredith, Fri Oct 19 16:54:36 EDT 2012
//   Added reverse connectivity (i.e. get cells attached to a node).
//
// ****************************************************************************
struct eavlRegularStructure
{
#define MAXDIM 4
    int dimension;
    int cellDims[MAXDIM];
    int nodeDims[MAXDIM];
#undef MAXDIM

    virtual string className() const {return "eavlRegularStructure";}
    virtual eavlStream& serialize(eavlStream &s) const
    {
	s << dimension;
	s << cellDims[0] << cellDims[1] << cellDims[2] << cellDims[3];
	s << nodeDims[0] << nodeDims[1] << nodeDims[2] << nodeDims[3];
	return s;
    }
    virtual eavlStream& deserialize(eavlStream &s)
    {
	s >> dimension;
	s >> cellDims[0] >> cellDims[1] >> cellDims[2] >> cellDims[3];
	s >> nodeDims[0] >> nodeDims[1] >> nodeDims[2] >> nodeDims[3];
	return s;
    }

    EAVL_HOSTONLY void SetCellDimension(int dim, int *dims)
    {
        switch (dim)
        {
          case 1: SetCellDimension1D(dims[0]); break;
          case 2: SetCellDimension2D(dims[0],dims[1]); break;
          case 3: SetCellDimension3D(dims[0],dims[1],dims[2]); break;
          default: THROW(eavlException, "Unexpected number of dimensions")
        }
    }
    EAVL_HOSTONLY void SetNodeDimension(int dim, int *dims)
    {
        switch (dim)
        {
          case 1: SetNodeDimension1D(dims[0]); break;
          case 2: SetNodeDimension2D(dims[0],dims[1]); break;
          case 3: SetNodeDimension3D(dims[0],dims[1],dims[2]); break;
          default: THROW(eavlException, "Unexpected number of dimensions")
        }
    }
    EAVL_HOSTONLY void SetCellDimension1D(int i)
    {
        dimension = 1;
        cellDims[0] = i;
        nodeDims[0] = i+1;
    }
    EAVL_HOSTONLY void SetNodeDimension1D(int i)
    {
        dimension = 1;
        nodeDims[0] = i;
        cellDims[0] = i-1;
    }

    EAVL_HOSTONLY void SetCellDimension2D(int i, int j)
    {
        dimension = 2;
        cellDims[0] = i;
        nodeDims[0] = i+1;
        cellDims[1] = j;
        nodeDims[1] = j+1;
    }
    EAVL_HOSTONLY void SetNodeDimension2D(int i, int j)
    {
        dimension = 2;
        nodeDims[0] = i;
        cellDims[0] = i-1;
        nodeDims[1] = j;
        cellDims[1] = j-1;
    }

    EAVL_HOSTONLY void SetCellDimension3D(int i, int j, int k)
    {
        dimension = 3;
        cellDims[0] = i;
        nodeDims[0] = i+1;
        cellDims[1] = j;
        nodeDims[1] = j+1;
        cellDims[2] = k;
        nodeDims[2] = k+1;
    }
    EAVL_HOSTONLY void SetNodeDimension3D(int i, int j, int k)
    {
        dimension = 3;
        nodeDims[0] = i;
        cellDims[0] = i-1;
        nodeDims[1] = j;
        cellDims[1] = j-1;
        nodeDims[2] = k;
        cellDims[2] = k-1;
    }
    

    EAVL_HOSTDEVICE int CalculateNodeIndexDivForDimension(int dim)
    {
        if (dim >= dimension)
            return 1;

        int div = 1;
        for (int d=0; d < dim; d++)
        {
            div *= nodeDims[d];
        }
        return div;
    }
    EAVL_HOSTDEVICE int CalculateNodeIndexModForDimension(int dim)
    {
        if (dim >= dimension)
            return -1;

        return nodeDims[dim];
    }

    //
    EAVL_HOSTDEVICE int CalculateCellIndex1D(int i)
    {
        return i;
    }
    EAVL_HOSTDEVICE int CalculateCellIndex2D(int i, int j)
    {
        return j * cellDims[0] + i;
    }
    EAVL_HOSTDEVICE int CalculateCellIndex3D(int i, int j, int k)
    {
        return (k * cellDims[1] + j) * cellDims[0] + i;
    }

    //
    EAVL_HOSTDEVICE int CalculateNodeIndex1D(int i)
    {
        return i;
    }
    EAVL_HOSTDEVICE int CalculateNodeIndex2D(int i, int j)
    {
        return j * nodeDims[0] + i;
    }
    EAVL_HOSTDEVICE int CalculateNodeIndex3D(int i, int j, int k)
    {
        return (k * nodeDims[1] + j) * nodeDims[0] + i;
    }

    //
    EAVL_HOSTDEVICE void CalculateLogicalCellIndices1D(int index, int &i)
    {
        i = index;
    }
    EAVL_HOSTDEVICE void CalculateLogicalNodeIndices1D(int index, int &i)
    {
        i = index;
    }

    EAVL_HOSTDEVICE void CalculateLogicalCellIndices2D(int index,
                                                           int &i, int &j)
    {
        j = index / cellDims[0];
        i = index % cellDims[0];
    }
    EAVL_HOSTDEVICE void CalculateLogicalNodeIndices2D(int index,
                                                           int &i, int &j)
    {
        j = index / nodeDims[0];
        i = index % nodeDims[0];
    }

    EAVL_HOSTDEVICE void CalculateLogicalCellIndices3D(int index,
                                                        int &i, int &j, int &k)
    {
        int cellDims01 = cellDims[0] * cellDims[1];
        k = index / cellDims01;
        int indexij = index % cellDims01;
        j = indexij / cellDims[0];
        i = indexij % cellDims[0];
    }
    EAVL_HOSTDEVICE void CalculateLogicalNodeIndices3D(int index,
                                                        int &i, int &j, int &k)
    {
        int nodeDims01 = nodeDims[0] * nodeDims[1];
        k = index / nodeDims01;
        int indexij = index % nodeDims01;
        j = indexij / nodeDims[0];
        i = indexij % nodeDims[0];
    }


    EAVL_HOSTDEVICE int GetNumCells()
    {
        if (dimension == 1)
            return cellDims[0];
        else if (dimension == 2)
            return cellDims[0]*cellDims[1];
        else if (dimension == 3)
            return cellDims[0]*cellDims[1]*cellDims[2];
        else
            return 0;
    }
    EAVL_HOSTDEVICE int GetNumNodes()
    {
        if (dimension == 1)
            return nodeDims[0];
        else if (dimension == 2)
            return nodeDims[0]*nodeDims[1];
        else if (dimension == 3)
            return nodeDims[0]*nodeDims[1]*nodeDims[2];
        else
            return 0;
    }
    EAVL_HOSTDEVICE int GetNumFaces()
    {
        if (dimension == 3)
        {
            int numXY = cellDims[0] * cellDims[1] * nodeDims[2];
            int numXZ = cellDims[0] * nodeDims[1] * cellDims[2];
            int numYZ = nodeDims[0] * cellDims[1] * cellDims[2];
            return numXY + numXZ + numYZ;
        }
        return 0;
    }
    EAVL_HOSTDEVICE int GetNumEdges()
    {
        if (dimension == 1)
        {
            return cellDims[0];
        }
        if (dimension == 2)
        {
            int numX = cellDims[0] * nodeDims[1];
            int numY = nodeDims[0] * cellDims[1];
            return numX + numY;
        }
        else if (dimension == 3)
        {
            int numX = cellDims[0] * nodeDims[1] * nodeDims[2];
            int numY = nodeDims[0] * cellDims[1] * nodeDims[2];
            int numZ = nodeDims[0] * nodeDims[1] * cellDims[2];
            return numX + numY + numZ;
        }
        return 0;
    }

    EAVL_HOSTDEVICE int GetCellNodes(int index, int &npts, int *pts)
    {
        // note: if isosurface is wrong, this (and getcelledges)
        // are likely culprits.
        if (dimension == 1)
        {
            npts = 2;
            pts[0] = index;
            pts[1] = index+1;
            return EAVL_BEAM;
        }
        else if (dimension == 2)
        {
            int i,j;
            CalculateLogicalCellIndices2D(index, i,j);
            npts = 4;
            pts[0] = CalculateNodeIndex2D(i, j);
            pts[1] = pts[0] + 1;
            pts[2] = pts[0] + nodeDims[0];
            pts[3] = pts[2] + 1;
            return EAVL_PIXEL;
        }
        else if (dimension == 3)
        {
            int i,j,k;
            CalculateLogicalCellIndices3D(index, i,j,k);
            npts = 8;
            pts[0] = CalculateNodeIndex3D(i, j, k);
            pts[1] = pts[0] + 1;
            pts[2] = pts[0] + nodeDims[0];
            pts[3] = pts[2] + 1;
            pts[4] = pts[0] + nodeDims[0]*nodeDims[1];
            pts[5] = pts[4] + 1;
            pts[6] = pts[4] + nodeDims[0];
            pts[7] = pts[6] + 1;
            return EAVL_VOXEL;            
        }
        npts = 0;
        return EAVL_OTHER;
    }

    EAVL_HOSTDEVICE int GetNodeCells(int index, int &ncells, int *cells)
    {
        ncells = 0;
        if (dimension == 1)
        {
            if (index > 0)
                cells[ncells++] = index - 1;
            if (index < nodeDims[0] - 1)
                cells[ncells++] = index;
            return EAVL_POINT;
        }
        else if (dimension == 2)
        {
            int i,j;
            CalculateLogicalNodeIndices2D(index, i,j);
            if (i > 0 && j > 0)
                cells[ncells++] = CalculateCellIndex2D(i-1, j-1);
            if (i < nodeDims[0]-1 && j > 0)
                cells[ncells++] = CalculateCellIndex2D(i  , j-1);
            if (i > 0 && j < nodeDims[1]-1)
                cells[ncells++] = CalculateCellIndex2D(i-1, j  );
            if (i < nodeDims[0]-1 && j < nodeDims[1]-1)
                cells[ncells++] = CalculateCellIndex2D(i  , j  );
            return EAVL_POINT;
        }
        else if (dimension == 3)
        {
            int i,j,k;
            CalculateLogicalNodeIndices3D(index, i,j,k);
            if (i > 0 && j > 0 && k > 0)
                cells[ncells++] = CalculateCellIndex3D(i-1, j-1, k-1);
            if (i < nodeDims[0]-1 && j > 0 && k > 0)
                cells[ncells++] = CalculateCellIndex3D(i  , j-1, k-1);
            if (i > 0 && j < nodeDims[1]-1 && k > 0)
                cells[ncells++] = CalculateCellIndex3D(i-1, j  , k-1);
            if (i < nodeDims[0]-1 && j < nodeDims[1]-1 && k > 0)
                cells[ncells++] = CalculateCellIndex3D(i  , j  , k-1);

            if (i > 0 && j > 0 && k < nodeDims[2]-1)
                cells[ncells++] = CalculateCellIndex3D(i-1, j-1, k);
            if (i < nodeDims[0]-1 && j > 0 && k < nodeDims[2]-1)
                cells[ncells++] = CalculateCellIndex3D(i  , j-1, k);
            if (i > 0 && j < nodeDims[1]-1 && k < nodeDims[2]-1)
                cells[ncells++] = CalculateCellIndex3D(i-1, j  , k);
            if (i < nodeDims[0]-1 && j < nodeDims[1]-1 && k < nodeDims[2]-1)
                cells[ncells++] = CalculateCellIndex3D(i  , j  , k);
            return EAVL_POINT;
        }
        return EAVL_OTHER;
    }

    EAVL_HOSTDEVICE int GetCellEdges(int index, int &nedges, int *edges)
    {
        if (dimension == 1)
        {
            nedges = 1;
            edges[0] = index;
            return EAVL_BEAM;
        }
        else if (dimension == 2)
        {
            int xc = cellDims[0];
            //int yc = cellDims[1];
            int xn = nodeDims[0];
            int yn = nodeDims[1];
            int numX = xc * yn;
            //int numY = yc * xn;

            int j = index / xc;
            int i = index % xc;

            nedges = 4;
            edges[0] = (j+0)*xc + i;
            edges[1] = numX + j*xn + (i+1);
            edges[2] = (j+1)*xc + i;
            edges[3] = numX + j*xn + (i+0);
            return EAVL_PIXEL;
        }
        else if (dimension == 3)
        {
            int xc = cellDims[0];
            int yc = cellDims[1];
            //int zc = cellDims[2];
            int xn = nodeDims[0];
            int yn = nodeDims[1];
            int zn = nodeDims[2];
            int numX = xc * yn * zn;
            int numY = xn * yc * zn;
            //int numZ = xc * yn * zc;

            int k = index / (xc*yc);
            int j = (index - k*xc*yc) / xc;
            int i = index % xc;

            nedges = 12;
            // z-min edges in clockwise order
            edges[0 ] =             (k+0)*xc*yn + (j+0)*xc + i; // x, y0-z0
            edges[1 ] = numX      + (k+0)*xn*yc + j*xn + (i+1); // y, x1-z0
            edges[2 ] =             (k+0)*xc*yn + (j+1)*xc + i; // x, y1-z0
            edges[3 ] = numX      + (k+0)*xn*yc + j*xn + (i+0); // y, x0-z0

            // z-max edges in clockwise order
            edges[4 ] =             (k+1)*xc*yn + (j+0)*xc + i; // x, y0-z1
            edges[5 ] = numX      + (k+1)*xn*yc + j*xn + (i+1); // y, x1-z1
            edges[6 ] =             (k+1)*xc*yn + (j+1)*xc + i; // x, y1-z1
            edges[7 ] = numX      + (k+1)*xn*yc + j*xn + (i+0); // y, x0-z1

            // z edges in staggered order
            edges[8 ] = numX+numY + k*xn*yn + (j+0)*xn + (i+0); // z, x0-y0
            edges[9 ] = numX+numY + k*xn*yn + (j+0)*xn + (i+1); // z, x1-y0
            edges[10] = numX+numY + k*xn*yn + (j+1)*xn + (i+0); // z, x0-y1
            edges[11] = numX+numY + k*xn*yn + (j+1)*xn + (i+1); // z, x1-y1
            return EAVL_VOXEL;            
        }
        nedges = 0;
        return EAVL_OTHER;
    }

    EAVL_HOSTDEVICE int GetCellFaces(int index, int &nfaces, int *faces)
    {
        ///\todo: the order here needs to match eavlCellComponents voxel faces
        if (dimension == 3)
        {
            int xc = cellDims[0];
            int yc = cellDims[1];
            int zc = cellDims[2];
            int xn = nodeDims[0];
            int yn = nodeDims[1];
            int zn = nodeDims[2];
            int numXY = xc * yc * zn;
            int numXZ = xc * yn * zc;
            //int numYZ = xn * yc * zc;

            int k = index / (xc*yc);
            int j = (index - k*xc*yc) / xc;
            int i = index % xc;

            nfaces = 6;
            // YZ faces
            faces[0] = numXY+numXZ + k*xn*yc + j*xn + (i+0);
            faces[1] = numXY+numXZ + k*xn*yc + j*xn + (i+1);
            // XZ faces
            faces[2] = numXY + k*xc*yn + (j+0)*xc + i;
            faces[3] = numXY + k*xc*yn + (j+1)*xc + i;
            // XY faces
            faces[4] = (k+0)*xc*yc + j*xc + i;
            faces[5] = (k+1)*xc*yc + j*xc + i;

            return EAVL_VOXEL;
        }
        nfaces = 0;
        return EAVL_OTHER;
    }

    EAVL_HOSTDEVICE int GetFaceNodes(int faceindex, int &npts, int *pts)
    {
        if (dimension == 3)
        {
            int xc = cellDims[0];
            int yc = cellDims[1];
            int zc = cellDims[2];
            int xn = nodeDims[0];
            int yn = nodeDims[1];
            int zn = nodeDims[2];
            int numXY = xc * yc * zn;
            int numXZ = xc * yn * zc;
            //int numYZ = xn * yc * zc;

            npts = 4;
            if (faceindex < numXY)
            {
                // XY
                int index = faceindex;
                int x = index % xc;
                int y = (index / xc) % yc;
                int z = index / (xc * yc);
                pts[0] = (z)*xn*yn + (y+0)*xn + (x+0);
                pts[1] = (z)*xn*yn + (y+0)*xn + (x+1);
                pts[2] = (z)*xn*yn + (y+1)*xn + (x+0);
                pts[3] = (z)*xn*yn + (y+1)*xn + (x+1);
            }
            else if (faceindex < numXY + numXZ)
            {
                // XZ
                int index = faceindex - numXY;
                int x = index % xc;
                int y = (index / xc) % yn;
                int z = index / (xc * yn);
                pts[0] = (z+0)*xn*yn + (y)*xn + (x+0);
                pts[1] = (z+1)*xn*yn + (y)*xn + (x+0);
                pts[2] = (z+0)*xn*yn + (y)*xn + (x+1);
                pts[3] = (z+1)*xn*yn + (y)*xn + (x+1);
            }
            else
            {
                // YZ
                int index = faceindex - numXY - numXZ;
                int x = index % xn;
                int y = (index / xn) % yc;
                int z = index / (xn * yc);
                pts[0] = (z+0)*xn*yn + (y+0)*xn + (x);
                pts[1] = (z+0)*xn*yn + (y+1)*xn + (x);
                pts[2] = (z+1)*xn*yn + (y+0)*xn + (x);
                pts[3] = (z+1)*xn*yn + (y+1)*xn + (x);
            }
            
            return EAVL_PIXEL;
        }
        npts = 0;
        return EAVL_OTHER;
    }
    EAVL_HOSTDEVICE int GetEdgeNodes(int edgeindex, int &npts, int *pts)
    {
        if (dimension == 1)
        {
            npts = 2;
            pts[0] = edgeindex;
            pts[1] = edgeindex+1;
            return EAVL_BEAM;
        }
        else if (dimension == 2)
        {
            int xc = cellDims[0];
            //int yc = cellDims[1];
            int xn = nodeDims[0];
            int yn = nodeDims[1];
            int numX = xc * yn;
            //int numY = yc * xn;

            int i, j;
            if (edgeindex < numX)
            {
                int index = edgeindex;
                int x = index % xc;
                int y = index / xc;
                i = y*xn + (x+0);
                j = y*xn + (x+1);
            }
            else
            {
                int index = edgeindex - numX;
                int x = index % xn;
                int y = index / xn;
                i = (y+0)*xn + x;
                j = (y+1)*xn + x;
            }

            npts = 2;
            pts[0] = i;
            pts[1] = j;
            return EAVL_BEAM;
        }
        else if (dimension == 3)
        {
            int xc = cellDims[0];
            int yc = cellDims[1];
            //int zc = cellDims[2];
            int xn = nodeDims[0];
            int yn = nodeDims[1];
            int zn = nodeDims[2];
            int numX = xc * yn * zn;
            int numY = xn * yc * zn;
            //int numZ = xc * yn * zc;
            int i, j;

            if (edgeindex < numX)
            {
                int index = edgeindex;
                int x = index % xc;
                int y = (index / xc) % yn;
                int z = index / (xc * yn);
                i = z*xn*yn + y*xn + (x+0);
                j = z*xn*yn + y*xn + (x+1);
            }
            else if (edgeindex < numX + numY)
            {
                int index = edgeindex - numX;
                int x = index % xn;
                int y = (index / xn) % yc;
                int z = index / (xn * yc);
                i = z*xn*yn + (y+0)*xn + x;
                j = z*xn*yn + (y+1)*xn + x;
            }
            else
            {
                int index = edgeindex - numX - numY;
                int x = index % xn;
                int y = (index / xn) % yn;
                int z = index / (xn * yn);
                i = (z+0)*xn*yn + y*xn + x;
                j = (z+1)*xn*yn + y*xn + x;
            }
            
            npts = 2;
            pts[0] = i;
            pts[1] = j;
            return EAVL_BEAM;
        }
        npts = 0;
        return EAVL_OTHER;
    }
};

struct eavlRegularConnectivity
{
    eavlRegularStructure structure;
    eavlTopology         connType;
    eavlRegularConnectivity(const eavlRegularStructure &rs,
                            eavlTopology ct)
        : structure(rs), connType(ct)
    {
    }
    eavlRegularConnectivity(const eavlRegularConnectivity &rc)
        : structure(rc.structure), connType(rc.connType)
    {
    }
    
    virtual string className() const {return "eavlRegularConnectivity";}
    virtual eavlStream& serialize(eavlStream &s) const
    {
	s << className();
	structure.serialize(s);
	s << connType;
	return s;
    }
    virtual eavlStream& deserialize(eavlStream &s)
    {
	cout<<"FIX_THIS: "<<__FILE__<<" "<<__LINE__<<endl;
	return s;
    }
    EAVL_HOSTDEVICE int GetShapeType(int) const
    {
        switch (structure.dimension)
        {
          case 1: return EAVL_BEAM;
          case 2: return EAVL_PIXEL;
          case 3: return EAVL_VOXEL;
        }
        return EAVL_OTHER;
    }
    EAVL_HOSTDEVICE int GetElementComponents(int index, int &npts, int *pts)
    {
        switch (connType)
        {
          case EAVL_NODES_OF_CELLS: return structure.GetCellNodes(index, npts, pts);
          case EAVL_NODES_OF_EDGES: return structure.GetEdgeNodes(index, npts, pts);
          case EAVL_NODES_OF_FACES: return structure.GetFaceNodes(index, npts, pts);
          case EAVL_CELLS_OF_NODES: return structure.GetNodeCells(index, npts, pts);
          case EAVL_EDGES_OF_CELLS: return structure.GetCellEdges(index, npts, pts);
          case EAVL_FACES_OF_CELLS: return structure.GetCellFaces(index, npts, pts);
        }
        return 0;
    }
};


#endif
