#ifndef MORTON_BVH_H
#define MORTON_BVH_H
#include <stdio.h>
#include "eavlArray.h"
#include "eavlException.h"
#include "eavlRTUtil.h"
#include "eavlFunctorArray.h"
class BVHSOA
{
 public:
  //arrays size of leafs + inner nodes (2n -1)
  //where n = numPrimitives
  eavlFloatArray  *xmin;
  eavlFloatArray  *ymin;
  eavlFloatArray  *zmin;
  eavlFloatArray  *xmax;
  eavlFloatArray  *ymax;
  eavlFloatArray  *zmax;

  eavlIntArray    *parent;

  //only inner nodes have children size = (n-1)
  eavlIntArray    *leftChild;
  eavlIntArray    *rightChild;

  //only the leafs need centroids and Ids
  eavlFloatArray  *centroidX;
  eavlFloatArray  *centroidY;
  eavlFloatArray  *centroidZ;
  eavlIntArray    *primId;
  

  eavlVector3      extentMin;
  eavlVector3      extentMax;  
  int              maxSize;
  int              numPrimitives;
  int              numInner;

  eavlArrayIndexer *leafIndexer;
  BVHSOA(const int &_numPrimitives) 
  {
    numPrimitives = _numPrimitives;
    numInner      = numPrimitives - 1;
    maxSize       = numInner + numPrimitives;
    xmin = NULL;
    ymin = NULL;
    zmin = NULL;
    xmax = NULL;
    ymax = NULL;
    zmax = NULL;
    centroidX  = NULL;
    centroidY  = NULL;
    centroidZ  = NULL;
    primId     = NULL;
    parent     = NULL;
    leftChild  = NULL;
    rightChild = NULL;
    leafIndexer = new eavlArrayIndexer(1,1e9, 1, numInner);
    if(numPrimitives == 1) printf("Danger: only one primitve\n");
    if(maxSize > 0)
    {
      //max Size all nodes
      xmin = new eavlFloatArray("xmin",1, maxSize);
      ymin = new eavlFloatArray("ymin",1, maxSize);
      zmin = new eavlFloatArray("zmin",1, maxSize);
      xmax = new eavlFloatArray("xmax",1, maxSize);
      ymax = new eavlFloatArray("ymax",1, maxSize);
      zmax = new eavlFloatArray("zmax",1, maxSize);

      parent = new eavlIntArray("",1, maxSize);
      //leafs only
      centroidX = new eavlFloatArray("",1, numPrimitives);
      centroidY = new eavlFloatArray("",1, numPrimitives);
      centroidZ = new eavlFloatArray("",1, numPrimitives);

      primId = new eavlIntArray("",1, numPrimitives);
      //inner nodes only
      leftChild  = new eavlIntArray("leftChild",1, numInner);
      rightChild = new eavlIntArray("rightChild",1, numInner);
      
    }
    else
    {
      THROW(eavlException, "BVHSOA size must be greater than zero.");
    }
  }

  void print()
  {
    cout<<"Total BBox: "<<extentMin<<" - "<<extentMax<<endl;
  }

  ~BVHSOA()
  {
    if(xmin != NULL)
    {
      delete xmin;
      delete ymin;
      delete zmin;
      delete xmax;
      delete ymax;
      delete zmax;
      delete centroidX;
      delete centroidY;
      delete centroidZ;
      delete primId;
      delete parent;
      delete rightChild;
      delete leftChild;
      delete leafIndexer;
    } 
  }          
};


class MortonBVHBuilder
{
  private:      
    bool             forceCpu;
    bool             convertedToAoS;
    bool             wasEavlArrayGiven;
    int              verbose;
    BVHSOA          *bvh;
    eavlIntArray    *indexes;
    eavlIntArray    *mortonCodes;
    eavlFloatArray  *tmpFloat;
    eavlIntArray    *tmpInt;
    eavlFloatArray  *innerNodes;
    eavlIntArray    *leafNodes;


    void findAABBs();
    void sort();
    void propagateAABBs();
    void flatten();         //Convert representation from SoA to AoS

  public:
    eavlFloatArray *verts;
    primitive_t    primitveType;
    int            numPrimitives;

    MortonBVHBuilder(eavlFloatArray * _verts, int _numPrimitives, primitive_t _primitveType);
    ~MortonBVHBuilder();
    void build();
    void setVerbose(const int &level);
    float * getInnerNodes(int &_size);
    int * getLeafNodes(int &_size);

    eavlFloatArray * getInnerNodes();
    eavlIntArray * getLeafNodes();
};
#endif