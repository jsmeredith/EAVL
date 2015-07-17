#include "eavlRayTriangleGeometry.h"
#include "eavlMatrix4x4.h"
#include "eavlMapOp.h"


//sven woop unit triangle intersection test 2004 PhD thesis page 35
struct WoopifyFunctor
{
    WoopifyFunctor(){}

    EAVL_FUNCTOR tuple<float,float,float,float,
                       float,float,float,float,
                       float,float,float,float> operator()(tuple<float,float,float,
                                                                 float,float,float,
                                                                 float,float,float> input){
       eavlVector3 a(get<0>(input),get<1>(input),get<2>(input));
       eavlVector3 b(get<3>(input),get<4>(input),get<5>(input));
       eavlVector3 c(get<6>(input),get<7>(input),get<8>(input));

       //get the two edge vectors 
       eavlVector3 e1 = a - c;
       eavlVector3 e2 = b - c;
       eavlVector3 n = e1 % e2;

       eavlMatrix4x4 mat(e1.x, e2.x,  n.x, c.x,
                         e1.y, e2.y,  n.y, c.y,
                         e1.z, e2.z,  n.z, c.z,
                             0,    0,    0, 1);
       //create the unit triangle transform
       mat.Invert();
        return tuple<float,float,float,float,
                     float,float,float,float,
                     float,float,float,float>(mat.m[2][0] ,mat.m[2][1], mat.m[2][2], -mat.m[2][3],
                                              mat.m[1][0] ,mat.m[1][1], mat.m[1][2],  mat.m[1][3],
                                              mat.m[0][0] ,mat.m[0][1], mat.m[0][2],  mat.m[0][3]);
        // return tuple<float,float,float,float,
        //              float,float,float,float,
        //              float,float,float,float>(0 ,0, 0, 0,
        //                                       0 ,0, 0,  0,
        //                                       0,0, 0,  0);
    }

};

void eavlRayTriangleGeometry::woopifyVerts(eavlFloatArray * _vertices, const int &_size)
{
    //eavlFloatArray *verts = new eavlFloatArray("vertsIn",1, _size * 9);
    //for(int i = 0; i < _size * 9; i++) verts->SetValue(i, _vertices[i]);
	eavlFloatArray *vertsOut =  new eavlFloatArray("", 1, _size * 12);
	eavlArrayIndexer v0Inx;
    eavlArrayIndexer v0Iny;
    eavlArrayIndexer v0Inz;
    eavlArrayIndexer v1Inx;
    eavlArrayIndexer v1Iny;
    eavlArrayIndexer v1Inz;
    eavlArrayIndexer v2Inx;
    eavlArrayIndexer v2Iny;
    eavlArrayIndexer v2Inz;

    v0Inx.add = 0;
    v0Iny.add = 1;
    v0Inz.add = 2;
    v1Inx.add = 3;
    v1Iny.add = 4;
    v1Inz.add = 5;
    v2Inx.add = 6;
    v2Iny.add = 7;
    v2Inz.add = 8;

    v0Inx.mul = 9;
    v0Iny.mul = 9;
    v0Inz.mul = 9;
    v1Inx.mul = 9;
    v1Iny.mul = 9;
    v1Inz.mul = 9;
    v2Inx.mul = 9;
    v2Iny.mul = 9;
    v2Inz.mul = 9;

    eavlArrayIndexer m20;
    eavlArrayIndexer m21;
    eavlArrayIndexer m22;
    eavlArrayIndexer m23;
    eavlArrayIndexer m10;
    eavlArrayIndexer m11;
    eavlArrayIndexer m12;
    eavlArrayIndexer m13;
    eavlArrayIndexer m00;
    eavlArrayIndexer m01;
    eavlArrayIndexer m02;
    eavlArrayIndexer m03;

    m20.add = 0;
    m21.add = 1;
    m22.add = 2;
    m23.add = 3;
    m10.add = 4;
    m11.add = 5;
    m12.add = 6;
    m13.add = 7;
    m00.add = 8;
    m01.add = 9;
    m02.add = 10;
    m03.add = 11;

    m20.mul = 12;
    m21.mul = 12;
    m22.mul = 12;
    m23.mul = 12;
    m10.mul = 12;
    m11.mul = 12;
    m12.mul = 12;
    m13.mul = 12;
    m00.mul = 12;
    m01.mul = 12;
    m02.mul = 12;
    m03.mul = 12;

    eavlExecutor::AddOperation(
        new_eavlMapOp(eavlOpArgs(eavlIndexable<eavlFloatArray>(_vertices, v0Inx),
                                 eavlIndexable<eavlFloatArray>(_vertices, v0Iny),
                                 eavlIndexable<eavlFloatArray>(_vertices, v0Inz),
                                 eavlIndexable<eavlFloatArray>(_vertices, v1Inx),
                                 eavlIndexable<eavlFloatArray>(_vertices, v1Iny),
                                 eavlIndexable<eavlFloatArray>(_vertices, v1Inz),
                                 eavlIndexable<eavlFloatArray>(_vertices, v2Inx),
                                 eavlIndexable<eavlFloatArray>(_vertices, v2Iny),
                                 eavlIndexable<eavlFloatArray>(_vertices, v2Inz)),
                      eavlOpArgs(eavlIndexable<eavlFloatArray>(vertsOut, m20),
                                 eavlIndexable<eavlFloatArray>(vertsOut, m21),
                                 eavlIndexable<eavlFloatArray>(vertsOut, m22),
                                 eavlIndexable<eavlFloatArray>(vertsOut, m23),
                                 eavlIndexable<eavlFloatArray>(vertsOut, m10),
                                 eavlIndexable<eavlFloatArray>(vertsOut, m11),
                                 eavlIndexable<eavlFloatArray>(vertsOut, m12),
                                 eavlIndexable<eavlFloatArray>(vertsOut, m13),
                                 eavlIndexable<eavlFloatArray>(vertsOut, m00),
                                 eavlIndexable<eavlFloatArray>(vertsOut, m01),
                                 eavlIndexable<eavlFloatArray>(vertsOut, m02),
                                 eavlIndexable<eavlFloatArray>(vertsOut, m03)),
                      WoopifyFunctor()),
                      "woopit");
        eavlExecutor::Go();

     vertices = new eavlTextureObject<float4>(vertsOut->GetNumberOfTuples() / 4, vertsOut, true);
}

