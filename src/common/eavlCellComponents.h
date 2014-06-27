// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_TRIANGULATION_TABLES_H
#define EAVL_TRIANGULATION_TABLES_H

// Note: isosurface tables have moved into the 'new' (gpu-friendly) flavor

extern signed char eavlTetEdges[6][2];
extern signed char eavlTetTriangleFaces[4][3];

extern signed char eavlPyramidEdges[8][2];
extern signed char eavlPyramidTriangleFaces[4][3];
extern signed char eavlPyramidQuadFaces[1][4];

extern signed char eavlWedgeEdges[9][2];
extern signed char eavlWedgeTriangleFaces[2][3];
extern signed char eavlWedgeQuadFaces[3][4];

extern signed char eavlHexEdges[12][2];
extern signed char eavlHexQuadFaces[6][4];

extern signed char eavlVoxEdges[12][2];
extern signed char eavlVoxQuadFaces[6][4];

extern signed char eavlTriEdges[3][2];

extern signed char eavlQuadEdges[4][2];

extern signed char eavlPixelEdges[4][2];

extern signed char eavlLineEdges[1][2];

#endif
