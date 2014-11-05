#ifndef MORTON_BVH_H
#define MORTON_BVH_H
#include <stdio.h>
#include "eavlArray.h"
#include "eavlException.h"
#include "eavlRTUtil.h"

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
    int              verbose;
    BVHSOA          *bvh;
    eavlIntArray    *indexes;
    eavlIntArray    *mortonCodes;
    eavlFloatArray  *tmpFloat;
    eavlIntArray    *tmpInt;
    eavlFloatArray  *innerNodes;
    eavlFloatArray  *leafNodes;


    void findAABBs();
    void sort();
    void propagateAABBs();
    

  public:
    float         *verts;
    primitive_t    primitveType;
    int            numPrimitives;

    MortonBVHBuilder(float * _verts, int _numPrimitives, primitive_t _primitveType)
      : verts(_verts), numPrimitives(_numPrimitives), primitveType(_primitveType)
    {

      verbose = 0;
      if(eavlExecutor::GetExecutionMode() == eavlExecutor::ForceCPU ) forceCpu = true;
      else forceCpu = false;

      if(numPrimitives < 1) THROW(eavlException, "Number of primitives must be greater that zero.");
      if(verts == NULL)     THROW(eavlException, "Verticies can't be NULL");
      //Insert preprocess that splits triangles before any of the memory is allocated
      bvh     = new BVHSOA(numPrimitives);
      indexes = new eavlIntArray("idx",1,numPrimitives);
      tmpInt  = new eavlIntArray("tmp",1,numPrimitives);
      //innerNodes  = new BVHInnerNodeSOA(numPrimitives - 1);
      //TODO: create indexing operation
      for(int i = 0; i < numPrimitives; i++) indexes->SetValue(i,i);

      mortonCodes = new eavlIntArray("mortonCodes",1,numPrimitives);

      tmpFloat   = new eavlFloatArray("tmpSpace",1, 2 * numPrimitives -1);
      //hand these arrays off to the consumer and let them deaL with deleting them.
      innerNodes = new eavlFloatArray("inner",1, (numPrimitives -1) * 16);  //16 flat values per node
      leafNodes  = new eavlFloatArray("leafs",1, numPrimitives * 2);
    }

    ~MortonBVHBuilder()
    {
      delete mortonCodes;
      delete bvh;
      delete indexes;
      delete tmpFloat;
      delete tmpInt;
    }

    void build();
    void setVerbose(const int &level);
    eavlFloatArray * getInnerNodes(){ return innerNodes; }
    eavlFloatArray * getLeafNodes(){ return leafNodes; }


};
#endif