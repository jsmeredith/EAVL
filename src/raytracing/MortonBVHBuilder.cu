#include "MortonBVHBuilder.h"
#include "eavlTextureObject.h"
#include "eavlReduceOp_1.h"
#include "eavlMapOp.h"
#include "eavlRadixSortOp.h"
#include "eavlCountingIterator.h"
#include "eavlGatherOp.h"
#include <algorithm>    
using namespace std; 

#ifdef HAVE_OPENMP
#include <omp.h>
#endif

//This is the structure of the flat BVH inner 
//layout
struct FlatIndxr
{
    eavlArrayIndexer xmin1;
    eavlArrayIndexer ymin1;
    eavlArrayIndexer zmin1;
    eavlArrayIndexer xmax1;
    eavlArrayIndexer ymax1;
    eavlArrayIndexer zmax1;
    eavlArrayIndexer xmin2;
    eavlArrayIndexer ymin2;
    eavlArrayIndexer zmin2;
    eavlArrayIndexer xmax2;
    eavlArrayIndexer ymax2;
    eavlArrayIndexer zmax2;

    eavlArrayIndexer lChild;
    eavlArrayIndexer rChild;

    FlatIndxr()
    {
        xmin1.add = 0;
        ymin1.add = 1;
        zmin1.add = 2;
        xmax1.add = 3;
        ymax1.add = 4;
        zmax1.add = 5;
        xmin2.add = 6;
        ymin2.add = 7;
        zmin2.add = 8;
        xmax2.add = 9;
        ymax2.add = 10;
        zmax2.add = 11;

        lChild.add = 12;
        rChild.add = 13;

        xmin1.mul = 16;
        ymin1.mul = 16;
        zmin1.mul = 16;
        xmax1.mul = 16;
        ymax1.mul = 16;
        zmax1.mul = 16;
        xmin2.mul = 16;
        ymin2.mul = 16;
        zmin2.mul = 16;
        xmax2.mul = 16;
        ymax2.mul = 16;
        zmax2.mul = 16;

        lChild.mul = 16;
        rChild.mul = 16;
        //two extra floats with padding not shown
    }
};

void validate(BVHSOA *bvh, int numLeafs, int currentNode, int &count)
{
    if(currentNode >= numLeafs - 1) 
    {
        count++;
        return; //at leaf: inc and get out
    }

    validate(bvh, numLeafs, bvh->leftChild->GetValue(currentNode), count);
    validate(bvh, numLeafs, bvh->rightChild->GetValue(currentNode), count);

}


MortonBVHBuilder::MortonBVHBuilder(eavlFloatArray* _verts, int _numPrimitives, primitive_t _primitveType)
  : verts(_verts), numPrimitives(_numPrimitives), primitveType(_primitveType)
{

      verbose = 0;
    
      convertedToAoS = false;
      wasEavlArrayGiven = false;
      if(numPrimitives < 1) THROW(eavlException, "Number of primitives must be greater that zero.");
      if(verts == NULL)     THROW(eavlException, "Verticies can't be NULL");
      //Insert preprocess that splits triangles before any of the memory is allocated
      bvh     = new BVHSOA(numPrimitives);
      indexes = new eavlIntArray("idx",1,numPrimitives);
      tmpInt  = new eavlIntArray("tmp",1,numPrimitives);
    
      eavlCountingIterator::generateIterator(indexes);

      mortonCodes = new eavlIntArray("mortonCodes",1,numPrimitives);

      tmpFloat   = new eavlFloatArray("tmpSpace",1, 2 * numPrimitives -1);
}

MortonBVHBuilder::~MortonBVHBuilder()
{
    delete mortonCodes;
    delete bvh;
    delete indexes;
    delete tmpFloat;
    delete tmpInt;
    if(!wasEavlArrayGiven) //we gave up control of this memory
    {
        delete leafNodes;
        delete innerNodes;    
    }
    
}
//TODO: this might work better with Global mem since only a few values are accessed


template<primitive_t primType> 
struct AABBFunctor
{ 
	eavlTextureObject<float4> verts;
	AABBFunctor(eavlTextureObject<float4> *_verts)
	: verts(*_verts)
	{}
	EAVL_FUNCTOR tuple<float, float, float, float, float, float> operator()(int idx)
	{
        float xmin;
        float xmax;
        float ymin;
        float ymax;
        float zmin;
        float zmax;

        if(primType == SPHERE)
        {
            float4 sdata = verts.getValue(idx);
            eavlVector3 temp(0,0,0);
            eavlVector3 center( sdata.x, sdata.y, sdata.z );
            
            float radius = sdata.w;
            temp.x = radius;
            temp.y = 0;
            temp.z = 0;

            eavlVector3 p = center + temp;
            //set first point to max and min
            xmin = p.x; xmax = p.x;
            ymin = p.y; ymax = p.y;
            zmin = p.z; zmax = p.z;

            p = center - temp;
            xmin = min(xmin, p.x); xmax = max(xmax, p.x);
            ymin = min(ymin, p.y); ymax = max(ymax, p.y);
            zmin = min(zmin, p.z); zmax = max(zmax, p.z);
            
            temp.x = 0;
            temp.y = radius;
            temp.z = 0;
            
            p = center + temp;
            xmin = min(xmin, p.x); xmax = max(xmax, p.x);
            ymin = min(ymin, p.y); ymax = max(ymax, p.y);
            zmin = min(zmin, p.z); zmax = max(zmax, p.z);

            p = center - temp;
            xmin = min(xmin, p.x); xmax = max(xmax, p.x);
            ymin = min(ymin, p.y); ymax = max(ymax, p.y);
            zmin = min(zmin, p.z); zmax = max(zmax, p.z);

            temp.x = 0;
            temp.y = 0;
            temp.z = radius;
            p = center + temp;
            xmin = min(xmin, p.x); xmax = max(xmax, p.x);
            ymin = min(ymin, p.y); ymax = max(ymax, p.y);
            zmin = min(zmin, p.z); zmax = max(zmax, p.z);

            p = center - temp;
            xmin = min(xmin, p.x); xmax = max(xmax, p.x);
            ymin = min(ymin, p.y); ymax = max(ymax, p.y);
            zmin = min(zmin, p.z); zmax = max(zmax, p.z);
        }

        if(primType == CYLINDER)
        {
            eavlVector3 temp(0,0,0);
            float4 c1 = verts.getValue(idx * 2);
            float4 c2 = verts.getValue(idx * 2 + 1);
            eavlVector3 base( c1.x, c1.y, c1.z );
            float radius = c1.w;
            eavlVector3 axis( c2.x, c2.y, c2.z );
            float height = c2.w;
            eavlVector3 top = base + axis * height;

            
            temp.x = radius;
            temp.y = 0;
            temp.z = 0;

            eavlVector3 p = base + temp;
            xmin = p.x; xmax = p.x;
            ymin = p.y; ymax = p.y;
            zmin = p.z; zmax = p.z;

            p = base - temp;
            xmin = min(xmin, p.x); xmax = max(xmax, p.x);
            ymin = min(ymin, p.y); ymax = max(ymax, p.y);
            zmin = min(zmin, p.z); zmax = max(zmax, p.z);

            p = top + temp;
            xmin = min(xmin, p.x); xmax = max(xmax, p.x);
            ymin = min(ymin, p.y); ymax = max(ymax, p.y);
            zmin = min(zmin, p.z); zmax = max(zmax, p.z);

            p = top - temp;
            xmin = min(xmin, p.x); xmax = max(xmax, p.x);
            ymin = min(ymin, p.y); ymax = max(ymax, p.y);
            zmin = min(zmin, p.z); zmax = max(zmax, p.z);

           
            temp.x = 0;
            temp.y = radius;
            temp.z = 0;
            p = base + temp;
            xmin = min(xmin, p.x); xmax = max(xmax, p.x);
            ymin = min(ymin, p.y); ymax = max(ymax, p.y);
            zmin = min(zmin, p.z); zmax = max(zmax, p.z);

            p = base - temp;
            xmin = min(xmin, p.x); xmax = max(xmax, p.x);
            ymin = min(ymin, p.y); ymax = max(ymax, p.y);
            zmin = min(zmin, p.z); zmax = max(zmax, p.z);

            p = top + temp;
            xmin = min(xmin, p.x); xmax = max(xmax, p.x);
            ymin = min(ymin, p.y); ymax = max(ymax, p.y);
            zmin = min(zmin, p.z); zmax = max(zmax, p.z);

            p = top - temp;
            xmin = min(xmin, p.x); xmax = max(xmax, p.x);
            ymin = min(ymin, p.y); ymax = max(ymax, p.y);
            zmin = min(zmin, p.z); zmax = max(zmax, p.z);

            temp.x = 0;
            temp.y = 0;
            temp.z = radius;
            
            p = base + temp;
            xmin = min(xmin, p.x); xmax = max(xmax, p.x);
            ymin = min(ymin, p.y); ymax = max(ymax, p.y);
            zmin = min(zmin, p.z); zmax = max(zmax, p.z);
            
            p = base - temp;
            xmin = min(xmin, p.x); xmax = max(xmax, p.x);
            ymin = min(ymin, p.y); ymax = max(ymax, p.y);
            zmin = min(zmin, p.z); zmax = max(zmax, p.z);

            p = top + temp;
            xmin = min(xmin, p.x); xmax = max(xmax, p.x);
            ymin = min(ymin, p.y); ymax = max(ymax, p.y);
            zmin = min(zmin, p.z); zmax = max(zmax, p.z);

            p = top - temp;
            xmin = min(xmin, p.x); xmax = max(xmax, p.x);
            ymin = min(ymin, p.y); ymax = max(ymax, p.y);
            zmin = min(zmin, p.z); zmax = max(zmax, p.z);
        }
    	

        return tuple<float, float, float, float, float, float>(xmin, ymin, zmin, xmax, ymax, zmax);
	} 
};


struct AABBTriFunctor
{ 
    eavlTextureObject<float> verts;
    AABBTriFunctor(eavlTextureObject<float> *_verts)
    : verts(*_verts)
    {}
    EAVL_FUNCTOR tuple<float, float, float, float, float, float> operator()(int idx)
    {
        float xmin;
        float xmax;
        float ymin;
        float ymax;
        float zmin;
        float zmax;

        eavlVector3 a,b,c;
        a.x = verts.getValue(idx * 9 + 0);
        a.y = verts.getValue(idx * 9 + 1);
        a.z = verts.getValue(idx * 9 + 2);
        b.x = verts.getValue(idx * 9 + 3);
        b.y = verts.getValue(idx * 9 + 4);
        b.z = verts.getValue(idx * 9 + 5);
        c.x = verts.getValue(idx * 9 + 6);
        c.y = verts.getValue(idx * 9 + 7);
        c.z = verts.getValue(idx * 9 + 8);

        xmin = min(a.x, min(b.x, c.x));
        xmax = max(a.x, max(b.x, c.x));
        ymin = min(a.y, min(b.y, c.y));
        ymax = max(a.y, max(b.y, c.y));
        zmin = min(a.z, min(b.z, c.z));
        zmax = max(a.z, max(b.z, c.z));

        return tuple<float, float, float, float, float, float>(xmin, ymin, zmin, xmax, ymax, zmax);
    } 
};

struct CentroidFunctor
{

	CentroidFunctor(){}
	EAVL_FUNCTOR tuple<float, float, float> operator()(tuple<float, float, float, float, float, float>  bbox)
	{												
		eavlVector3 minPoint(get<0>(bbox),get<1>(bbox),get<2>(bbox));
		eavlVector3 maxPoint(get<3>(bbox),get<4>(bbox),get<5>(bbox));

		eavlVector3 dir = maxPoint - minPoint;

		float halfDist = sqrt(dir * dir) * 0.5f;

		dir.normalize();

		eavlVector3 centroid = minPoint + halfDist * dir;  

        return tuple<float, float, float>(centroid.x, centroid.y, centroid.z);
	} 
};

struct MortonFunctor
{
	eavlVector3 mins;
	eavlVector3 invExtent;
	MortonFunctor(const eavlVector3 &mn, const eavlVector3 &mx)
	: mins(mn)
	{
		invExtent = mx - mn;
		invExtent.x = (invExtent.x == 0) ? 0 : 1.f / invExtent.x;
		invExtent.y = (invExtent.y == 0) ? 0 : 1.f / invExtent.y;
		invExtent.z = (invExtent.z == 0) ? 0 : 1.f / invExtent.z;
	}
	EAVL_FUNCTOR tuple<int> operator()(tuple<float, float, float>  input)
	{												
		eavlVector3 centroid(get<0>(input),get<1>(input),get<2>(input));
		//normalize to the unit cube
		centroid -= mins;
		centroid.x = centroid.x * invExtent.x;
		centroid.y = centroid.y * invExtent.y;
		centroid.z = centroid.z * invExtent.z;

		unsigned int code = morton3D(centroid.x, centroid.y, centroid.z); 
        return tuple<int>(code);
	} 
};

struct TreeFunctor
{ 
	int leafCount;
    int innerCount;

	eavlTextureObject<unsigned int> mortonCodes;
    eavlFunctorArray<int> parents;


	TreeFunctor(eavlTextureObject<unsigned int> *codes, 
                int _leafCount, 
                eavlFunctorArray<int> par)
	: mortonCodes(*codes), leafCount(_leafCount), parents(par)
	{
        innerCount = leafCount - 1;
    }

	/**
	 * returns the count of largest shared prefix between
	 * two morton codes. Ties are broken by the indexes
	 * a and b.
	 * @param  a - index of value one
	 * @param  b - index of value two
	 * @return count of the largest binary prefix 
	 */
	EAVL_HOSTDEVICE int cclz(unsigned int &x)
	{
	  unsigned int y;
	  int n = 32;
	  y = x >>16; if (y != 0) { n = n -16; x = y; }
    y = x >> 8; if (y != 0) { n = n - 8; x = y; }
    y = x >> 4; if (y != 0) { n = n - 4; x = y; }
    y = x >> 2; if (y != 0) { n = n - 2; x = y; }
    y = x >> 1; if (y != 0) return n - 2;
    return n - x;
	}
	EAVL_HOSTDEVICE int delta(const int &a, const int &b)
	{
		bool tie = false;
		bool outOfRange = (b < 0 || b > leafCount -1);
        int bb = (outOfRange) ? 0 : b; //still make the call but with a valid adderss
		unsigned int aCode =  mortonCodes.getValue(a);
		unsigned int bCode =  mortonCodes.getValue(bb);
		unsigned int exOr = aCode ^ bCode; //use xor to find where they differ
		tie = (exOr == 0);
		exOr = tie ? a ^ bb : exOr; //break the tie, a and b will always differ 
#ifdef __CUDA_ARCH__
    int count = clz(exOr);
#else 
    //
    //  Aparently on Surface@llnl this doesn't work.
    //
    int count = cclz(exOr);
#endif
		if(tie) count += 32; 
		count = (outOfRange) ? -1 : count;
		return count;
	}

	EAVL_FUNCTOR tuple<int, int> operator()(int idx)
	{								
        if(idx > leafCount - 2) return tuple<int, int>(-1,-1);					
        //determine range direction
        int d = 0 > (delta(idx, idx + 1) - delta(idx, idx - 1)) ?  -1 : 1;
        
        //find upper bound for the length of the range
        int minDelta = delta(idx, idx - d);
        int lMax = 2;
        while( delta(idx, idx + lMax * d) > minDelta ) lMax *= 2; 

        //binary search to find the lower bound
        int l = 0;
        for(int t = lMax / 2; t >= 1; t/=2) 
        {
            if(delta(idx, idx + (l + t)*d ) > minDelta) l += t;
        }

        int j = idx + l * d;
        int deltaNode = delta(idx,j);
        int s = 0;
        float divFactor = 2.f; 
        //find the split postition using a binary search
        for(int t = (int) ceil(l / divFactor);; divFactor*=2, t = (int) ceil(l / divFactor) )
        {    
            if(delta(idx, idx + (s + t) * d) > deltaNode)
            {
                s += t;
            } 
            
            if(t == 1) break;
        }

        int split = idx + s * d + min(d,0);
        int leftChild, rightChild;
        //assign parent/child pointers
        if(min(idx, j) == split)
        {
          //leaf
          parents[split + innerCount] = idx;
          leftChild = split + innerCount;
        } 
        else
        {
            parents[split] = idx;
            leftChild = split;        //inner node   
        }
         

        if(max(idx, j) == split + 1)
        {
           //leaf
          parents[split + innerCount + 1] = idx;
          rightChild = split + innerCount + 1;  
        } 
        else
        {
            parents[split + 1] = idx;
            rightChild = split + 1;       
        }
        return tuple<int, int>(leftChild,rightChild);
	} 
};

struct BottomUpFunctor
{
    eavlFunctorArray<int> nodeCounters;
    int numLeafs;
    eavlFunctorArray<float> xmins;
    eavlFunctorArray<float> ymins;
    eavlFunctorArray<float> zmins;
    eavlFunctorArray<float> xmaxs;
    eavlFunctorArray<float> ymaxs;
    eavlFunctorArray<float> zmaxs;
    eavlFunctorArray<int>   lChild;
    eavlFunctorArray<int>   rChild;
    eavlFunctorArray<int>   parents;

    BottomUpFunctor(eavlFunctorArray<float> _xmins, 
                    eavlFunctorArray<float> _ymins, 
                    eavlFunctorArray<float> _zmins,
                    eavlFunctorArray<float> _xmaxs, 
                    eavlFunctorArray<float> _ymaxs, 
                    eavlFunctorArray<float> _zmaxs,
                    eavlFunctorArray<int> _lChild, 
                    eavlFunctorArray<int> _rChild,
                    eavlFunctorArray<int> _parents,
                    eavlFunctorArray<int> aCounters, int _numLeafs)
    :  xmins(_xmins), ymins(_ymins), zmins(_zmins),
       xmaxs(_xmaxs), ymaxs(_ymaxs), zmaxs(_zmaxs),
       lChild(_lChild), rChild(_rChild), parents(_parents),
       nodeCounters(aCounters), numLeafs(_numLeafs)
    {
        
    }

    EAVL_HOSTDEVICE bool checkIdx(int idx)
    {
        //if(idx < 0 || idx >= (numLeafs - 1 )) printf("ILLEGAL %d ", idx);
        int old;
        bool kill = false;
#ifdef __CUDA_ARCH__
        old = atomicAdd(&nodeCounters[idx],1);   
#else 
        #pragma omp atomic capture
        old = nodeCounters[idx]++;
#endif
        if(old == 0) kill = true;
        // if(kill) printf("Idx %d \n", idx);
        return kill;
    }


    EAVL_FUNCTOR tuple<int> operator()(int idx)
    {   
        //start traversal at the leaves                                            
        int node = idx + numLeafs - 1;
        do
        {
            //Go up one level to make sure everyone has children
            node = parents[node];
            //First thread to get to the node terminates,
            //Second  thread processes AABB of the children
            if(checkIdx(node)) return tuple<int>(idx);
            xmins[node] = min(xmins[lChild[node]], xmins[rChild[node]]);
            ymins[node] = min(ymins[lChild[node]], ymins[rChild[node]]);
            zmins[node] = min(zmins[lChild[node]], zmins[rChild[node]]);
            xmaxs[node] = max(xmaxs[lChild[node]], xmaxs[rChild[node]]);
            ymaxs[node] = max(ymaxs[lChild[node]], ymaxs[rChild[node]]);
            zmaxs[node] = max(zmaxs[lChild[node]], zmaxs[rChild[node]]);
        } while (node != 0);
        
        ;//printf("There can only be one! %d\n", idx);

        return tuple<int>(idx); //indexed to same value / Do nothing
    } 
};


struct InnerToFlatFunctor
{
    eavlFunctorArray<float> xmins;
    eavlFunctorArray<float> ymins;
    eavlFunctorArray<float> zmins;
    eavlFunctorArray<float> xmaxs;
    eavlFunctorArray<float> ymaxs;
    eavlFunctorArray<float> zmaxs;
    int    numPrimitives;
    int    primOffset;
    InnerToFlatFunctor(eavlFunctorArray<float> _xmins, 
                       eavlFunctorArray<float> _ymins, 
                       eavlFunctorArray<float> _zmins,
                       eavlFunctorArray<float> _xmaxs, 
                       eavlFunctorArray<float> _ymaxs, 
                       eavlFunctorArray<float> _zmaxs,
                       int _numPrims) 
                       : xmins(_xmins), ymins(_ymins), zmins(_zmins),
                         xmaxs(_xmaxs), ymaxs(_ymaxs), zmaxs(_zmaxs),
                         numPrimitives(_numPrims)
    {
        primOffset = numPrimitives - 1;
    }
    EAVL_FUNCTOR   tuple<float, float, float, float, float, float, float, float, float, float, float, float, float, float>
        operator()(tuple<int, int>  input)
    {                                               
        int lChild = get<0>(input);
        int rChild = get<1>(input);
        //Child nodes are indexed with neg numbers in this format
        //First prim is at -1
        int clIdx  = (lChild < numPrimitives - 1) ? lChild * 4 : -(lChild - primOffset) * 2 - 1;  
        int crIdx  = (rChild < numPrimitives - 1) ? rChild * 4 : -(rChild - primOffset) * 2 - 1;  
        //to avoid truncating interger value when casting to floating point
        float clf,crf;
        memcpy(&clf,&clIdx, 4);
        memcpy(&crf,&crIdx, 4);
        //each node stores the bounding boxes of its children 
        return tuple<float, float, float, float, float, float, float, float, float, float, float, float, float, float>
        (xmins[lChild], ymins[lChild], zmins[lChild], xmaxs[lChild], ymaxs[lChild], zmaxs[lChild],
         xmins[rChild], ymins[rChild], zmins[rChild], xmaxs[rChild], ymaxs[rChild], zmaxs[rChild],
         clf, crf);
    } 
};


struct LeafToFlatFunctor
{
    LeafToFlatFunctor(){}
    EAVL_FUNCTOR tuple<int, int> operator()(int  primId)
    {                                               
        return tuple<int, int>(1, primId);
    } 
};


void MortonBVHBuilder::setVerbose(const int &level)
{
	if(level > 0) verbose = level;
}

void MortonBVHBuilder::findAABBs()
{
    eavlTextureObject<float> *floatVerts = NULL;
    eavlTextureObject<float4> *float4Verts = NULL;
    //load verts into texture for bbox calculation
    //TODO:Does this make sense to have this as texture? 
    //3 reads per thread not really streaming many addresses
    if(primitveType == TRIANGLE)
    {
        floatVerts = new eavlTextureObject<float>(numPrimitives * 9, verts, false);
    }
    else if(primitveType == SPHERE)
    {
        float4Verts = new eavlTextureObject<float4>(numPrimitives, (float4*)verts, false);
    }
    else if(primitveType == CYLINDER)
    {
        float4Verts = new eavlTextureObject<float4>(numPrimitives * 2, (float4*)verts, false);
    }
    

	//calculate the AABBs of all the primitives
    if(primitveType == TRIANGLE)
    {
        eavlExecutor::AddOperation(
        new_eavlMapOp(eavlOpArgs(indexes),
                      eavlOpArgs(eavlIndexable<eavlFloatArray>(bvh->xmin, *bvh->leafIndexer),
                                 eavlIndexable<eavlFloatArray>(bvh->ymin, *bvh->leafIndexer),
                                 eavlIndexable<eavlFloatArray>(bvh->zmin, *bvh->leafIndexer),
                                 eavlIndexable<eavlFloatArray>(bvh->xmax, *bvh->leafIndexer),
                                 eavlIndexable<eavlFloatArray>(bvh->ymax, *bvh->leafIndexer),
                                 eavlIndexable<eavlFloatArray>(bvh->zmax, *bvh->leafIndexer)),
                      AABBTriFunctor(floatVerts)),
                      "AABB");
        eavlExecutor::Go();
    }
    else if(primitveType == SPHERE)
    {
        eavlExecutor::AddOperation(
        new_eavlMapOp(eavlOpArgs(indexes),
                      eavlOpArgs(eavlIndexable<eavlFloatArray>(bvh->xmin, *bvh->leafIndexer),
                                 eavlIndexable<eavlFloatArray>(bvh->ymin, *bvh->leafIndexer),
                                 eavlIndexable<eavlFloatArray>(bvh->zmin, *bvh->leafIndexer),
                                 eavlIndexable<eavlFloatArray>(bvh->xmax, *bvh->leafIndexer),
                                 eavlIndexable<eavlFloatArray>(bvh->ymax, *bvh->leafIndexer),
                                 eavlIndexable<eavlFloatArray>(bvh->zmax, *bvh->leafIndexer)),
                      AABBFunctor<SPHERE>(float4Verts)),
                      "AABB");
        eavlExecutor::Go();
    }
    else if(primitveType == CYLINDER)
    {
        eavlExecutor::AddOperation(
        new_eavlMapOp(eavlOpArgs(indexes),
                      eavlOpArgs(eavlIndexable<eavlFloatArray>(bvh->xmin, *bvh->leafIndexer),
                                 eavlIndexable<eavlFloatArray>(bvh->ymin, *bvh->leafIndexer),
                                 eavlIndexable<eavlFloatArray>(bvh->zmin, *bvh->leafIndexer),
                                 eavlIndexable<eavlFloatArray>(bvh->xmax, *bvh->leafIndexer),
                                 eavlIndexable<eavlFloatArray>(bvh->ymax, *bvh->leafIndexer),
                                 eavlIndexable<eavlFloatArray>(bvh->zmax, *bvh->leafIndexer)),
                      AABBFunctor<CYLINDER>(float4Verts)),
                      "AABB");
        eavlExecutor::Go();
    }
	
    if(floatVerts  != NULL) delete floatVerts;
    if(float4Verts != NULL) delete float4Verts;

    //we have to create the legacy indexer since Reduce is old.
    //It will also never be updated.
    eavlArrayWithLinearIndex lIndexer;
    lIndexer.div = 1;
    lIndexer.mul = 1;
    lIndexer.mod = INT_MAX;
    lIndexer.add = numPrimitives - 1; //leaf offset
    lIndexer.array = bvh->xmin;


    //create an array to store the reduction into
    eavlFloatArray *value = new eavlFloatArray("",1,1);
    //find the min and the max extents for each coordinate;
    //min
    eavlExecutor::AddOperation(
        new eavlReduceOp_1<eavlMinFunctor<float> >
        (lIndexer, value, eavlMinFunctor<float>(), numPrimitives), "min");

    eavlExecutor::Go();

    bvh->extentMin.x = value->GetValue(0);
    lIndexer.array = bvh->ymin;
    eavlExecutor::AddOperation(
        new eavlReduceOp_1<eavlMinFunctor<float> >
        (lIndexer, value, eavlMinFunctor<float>(), numPrimitives), "min");

    eavlExecutor::Go();

    bvh->extentMin.y = value->GetValue(0);
    lIndexer.array = bvh->zmin;
    eavlExecutor::AddOperation(
        new eavlReduceOp_1<eavlMinFunctor<float> >
        (lIndexer, value, eavlMinFunctor<float>(), numPrimitives), "min");

    eavlExecutor::Go();

    bvh->extentMin.z = value->GetValue(0);
    //max
    lIndexer.array = bvh->xmax;
    eavlExecutor::AddOperation(
        new eavlReduceOp_1<eavlMaxFunctor<float> >
        (lIndexer, value, eavlMaxFunctor<float>(), numPrimitives), "max");

    eavlExecutor::Go();

    bvh->extentMax.x = value->GetValue(0);
    lIndexer.array = bvh->ymax;
    eavlExecutor::AddOperation(
        new eavlReduceOp_1<eavlMaxFunctor<float> >
        (lIndexer, value, eavlMaxFunctor<float>(), numPrimitives), "max");

    eavlExecutor::Go();

    bvh->extentMax.y = value->GetValue(0);
    lIndexer.array = bvh->zmax;
    eavlExecutor::AddOperation(
        new eavlReduceOp_1<eavlMaxFunctor<float> >
        (lIndexer, value, eavlMaxFunctor<float>(), numPrimitives), "max");

    eavlExecutor::Go();

    bvh->extentMax.z = value->GetValue(0);

    delete value;
    if(verbose > 0) bvh->print();

    eavlExecutor::AddOperation(
    	new_eavlMapOp(eavlOpArgs(eavlIndexable<eavlFloatArray>(bvh->xmin, *bvh->leafIndexer),
                                 eavlIndexable<eavlFloatArray>(bvh->ymin, *bvh->leafIndexer),
                                 eavlIndexable<eavlFloatArray>(bvh->zmin, *bvh->leafIndexer),
                                 eavlIndexable<eavlFloatArray>(bvh->xmax, *bvh->leafIndexer),
                                 eavlIndexable<eavlFloatArray>(bvh->ymax, *bvh->leafIndexer),
                                 eavlIndexable<eavlFloatArray>(bvh->zmax, *bvh->leafIndexer)),
                      eavlOpArgs(bvh->centroidX, bvh->centroidY, bvh->centroidZ), 
                      CentroidFunctor(), numPrimitives ),
                      "Centroid");
    eavlExecutor::Go();
}

void MortonBVHBuilder::sort()
{
    // primitive ids and scatter indexes are the same
	eavlIntArray *idx = bvh->primId;
    eavlExecutor::AddOperation(
		new_eavlRadixSortOp(eavlOpArgs(mortonCodes),
                            eavlOpArgs(idx), true),
                            "Radix");
    eavlExecutor::Go();
    int tgather;

    /**
     * Allocation takes a long time, so 
     * we keep a temp array around to avoid 
     * that cost. Just do some pointer swapping.
     */
    eavlFloatArray *tmpPtr = NULL;
    if(verbose > 0) tgather = eavlTimer::Start();

    eavlExecutor::AddOperation(
        new_eavlGatherOp(eavlOpArgs(eavlIndexable<eavlFloatArray>(bvh->xmin, *bvh->leafIndexer)),
                         eavlOpArgs(eavlIndexable<eavlFloatArray>(tmpFloat,  *bvh->leafIndexer)),
                         eavlOpArgs(idx), numPrimitives),
                         "sorting");
    eavlExecutor::Go();
    tmpPtr = bvh->xmin;
    bvh->xmin = tmpFloat;
    tmpFloat = tmpPtr;

    eavlExecutor::AddOperation(
        new_eavlGatherOp(eavlOpArgs(eavlIndexable<eavlFloatArray>(bvh->ymin, *bvh->leafIndexer)),
                         eavlOpArgs(eavlIndexable<eavlFloatArray>(tmpFloat,  *bvh->leafIndexer)),
                         eavlOpArgs(idx), numPrimitives),
                         "sorting");
    eavlExecutor::Go();
    tmpPtr = bvh->ymin;
    bvh->ymin = tmpFloat;
    tmpFloat = tmpPtr;

    eavlExecutor::AddOperation(
        new_eavlGatherOp(eavlOpArgs(eavlIndexable<eavlFloatArray>(bvh->zmin, *bvh->leafIndexer)),
                         eavlOpArgs(eavlIndexable<eavlFloatArray>(tmpFloat,  *bvh->leafIndexer)),
                         eavlOpArgs(idx), numPrimitives),
                         "sorting");
    eavlExecutor::Go();
    tmpPtr = bvh->zmin;
    bvh->zmin = tmpFloat;
    tmpFloat = tmpPtr;

    eavlExecutor::AddOperation(
        new_eavlGatherOp(eavlOpArgs(eavlIndexable<eavlFloatArray>(bvh->xmax, *bvh->leafIndexer)),
                         eavlOpArgs(eavlIndexable<eavlFloatArray>(tmpFloat,  *bvh->leafIndexer)),
                         eavlOpArgs(idx), numPrimitives),
                         "sorting");
    eavlExecutor::Go();
    tmpPtr = bvh->xmax;
    bvh->xmax = tmpFloat;
    tmpFloat = tmpPtr;

    eavlExecutor::AddOperation(
        new_eavlGatherOp(eavlOpArgs(eavlIndexable<eavlFloatArray>(bvh->ymax, *bvh->leafIndexer)),
                         eavlOpArgs(eavlIndexable<eavlFloatArray>(tmpFloat,  *bvh->leafIndexer)),
                         eavlOpArgs(idx), numPrimitives),
                         "sorting");
    eavlExecutor::Go();
    tmpPtr = bvh->ymax;
    bvh->ymax = tmpFloat;
    tmpFloat = tmpPtr;

    eavlExecutor::AddOperation(
        new_eavlGatherOp(eavlOpArgs(eavlIndexable<eavlFloatArray>(bvh->zmax, *bvh->leafIndexer)),
                         eavlOpArgs(eavlIndexable<eavlFloatArray>(tmpFloat,  *bvh->leafIndexer)),
                         eavlOpArgs(idx), numPrimitives),
                         "sorting");
    eavlExecutor::Go();
    tmpPtr = bvh->zmax;
    bvh->zmax = tmpFloat;
    tmpFloat = tmpPtr;

    if(verbose > 0) cout<<"GATHER   RUNTIME: "<<eavlTimer::Stop(tgather,"rf")<<endl;
}

void MortonBVHBuilder::propagateAABBs()
{
    eavlExecutor::AddOperation(
        new_eavlMapOp(eavlOpArgs(mortonCodes),
                      eavlOpArgs(mortonCodes),
                      IntMemsetFunctor(0)), "");
    eavlExecutor::Go();

    eavlFunctorArray<int>   atomicCounters(mortonCodes);
    eavlFunctorArray<int>   parents(bvh->parent);
    eavlFunctorArray<int>   lChild(bvh->leftChild);
    eavlFunctorArray<int>   rChild(bvh->rightChild);

    eavlFunctorArray<float> xmins(bvh->xmin);
    eavlFunctorArray<float> ymins(bvh->ymin);
    eavlFunctorArray<float> zmins(bvh->zmin);
    eavlFunctorArray<float> xmaxs(bvh->xmax);
    eavlFunctorArray<float> ymaxs(bvh->ymax);
    eavlFunctorArray<float> zmaxs(bvh->zmax);

    eavlExecutor::AddOperation(
        new_eavlMapOp(eavlOpArgs(indexes),
                      eavlOpArgs(indexes), //dummy
                      BottomUpFunctor(xmins, ymins, zmins,
                                      xmaxs, ymaxs, zmaxs,
                                      lChild, rChild, parents,
                                      atomicCounters, numPrimitives)), "");
    eavlExecutor::Go();

}

void MortonBVHBuilder::build()
{

	int taabb;
    if(verbose > 0) taabb = eavlTimer::Start();
    //Calculate AABBs and centroids of the primitves
	findAABBs();
	if(verbose > 0) cout<<"AABB     RUNTIME: "<<eavlTimer::Stop(taabb,"rf")<<endl;
     //cout<<"VVVVVV "<<bvh->xmin->GetValue(bvh->numInner)<< " "<<bvh->zmax->GetValue(bvh->numInner)<<endl;

	//Generate Morton code based on the centriod of the AABB
	eavlExecutor::AddOperation(
    	new_eavlMapOp(eavlOpArgs(bvh->centroidX, bvh->centroidY, bvh->centroidZ),
                      eavlOpArgs(mortonCodes),
                      MortonFunctor(bvh->extentMin, bvh->extentMax)),
                      "Morton");
    eavlExecutor::Go();
     //cout<<"VVVVVV "<<bvh->xmin->GetValue(bvh->numInner)<< " "<<bvh->zmax->GetValue(bvh->numInner)<<endl;
    int tsort;
    if(verbose > 0) tsort = eavlTimer::Start();
    sort();
    if(verbose > 0) cout<<"SORT     RUNTIME: "<<eavlTimer::Stop(tsort,"rf")<<endl;

    eavlTextureObject<unsigned int> *mortonTexture = NULL;
    mortonTexture = new eavlTextureObject<unsigned int>( numPrimitives, 
                                                         mortonCodes,
                                                         false);
    //cout<<"VVVVVV "<<bvh->xmin->GetValue(bvh->numInner)<< " "<<bvh->zmax->GetValue(bvh->numInner)<<endl;
    //bvh->parent->SetValue(0,-1);
    eavlFunctorArray<int> parents(bvh->parent);

    //Build the tree in place. TODO: figure out a better way to set parent pointers
    // Current method will fail if the GPU falls back to the CPU
    int ttree;
    if(verbose > 0) ttree = eavlTimer::Start();
    eavlExecutor::AddOperation(
        new_eavlMapOp(eavlOpArgs(indexes),
                      eavlOpArgs(bvh->leftChild, bvh->rightChild),
                      TreeFunctor(mortonTexture, numPrimitives,parents), numPrimitives - 1),
                      "tree");
    eavlExecutor::Go();
    if(verbose > 0) cout<<"TREE     RUNTIME: "<<eavlTimer::Stop(ttree,"rf")<<endl;
    int tprop;
    if(verbose > 0) tprop = eavlTimer::Start();
    propagateAABBs();
    if(verbose > 0) cout<<"PROP     RUNTIME: "<<eavlTimer::Stop(tprop,"rf")<<endl;
    
    delete mortonTexture;
    //int count = 0;
    //validate(bvh, numPrimitives, 0, count);
    //if(count != numPrimitives) cout<<"BBBBBBBAADD "<<count<<endl;
    //cout<<bvh->xmin->GetValue(0)<<" "<<bvh->xmax->GetValue(0)<<endl;
}

void MortonBVHBuilder::flatten()
{
    //hand these arrays off to the consumer and let them deaL with deleting them.
    innerNodes = new eavlFloatArray("inner",1, (numPrimitives -1) * 16);  //16 flat values per node
    leafNodes  = new eavlIntArray("leafs",1, numPrimitives * 2);

    eavlFunctorArray<int>   atomicCounters(mortonCodes);
    eavlFunctorArray<int>   parents(bvh->parent);
    eavlFunctorArray<int>   lChild(bvh->leftChild);
    eavlFunctorArray<int>   rChild(bvh->rightChild);

    eavlFunctorArray<float> xmins(bvh->xmin);
    eavlFunctorArray<float> ymins(bvh->ymin);
    eavlFunctorArray<float> zmins(bvh->zmin);
    eavlFunctorArray<float> xmaxs(bvh->xmax);
    eavlFunctorArray<float> ymaxs(bvh->ymax);
    eavlFunctorArray<float> zmaxs(bvh->zmax);

    FlatIndxr flatIdx;
    //write out the array in parallel
    eavlExecutor::AddOperation(
        new_eavlMapOp(eavlOpArgs(innerNodes),
                      eavlOpArgs(innerNodes),
                      FloatMemsetFunctor(0)), "");
    eavlExecutor::Go();
    eavlExecutor::AddOperation(
        new_eavlMapOp(eavlOpArgs(bvh->leftChild, bvh->rightChild),
                      eavlOpArgs(eavlIndexable<eavlFloatArray>(innerNodes,  flatIdx.xmin1),
                                 eavlIndexable<eavlFloatArray>(innerNodes,  flatIdx.ymin1),
                                 eavlIndexable<eavlFloatArray>(innerNodes,  flatIdx.zmin1),
                                 eavlIndexable<eavlFloatArray>(innerNodes,  flatIdx.xmax1),
                                 eavlIndexable<eavlFloatArray>(innerNodes,  flatIdx.ymax1),
                                 eavlIndexable<eavlFloatArray>(innerNodes,  flatIdx.zmax1),
                                 eavlIndexable<eavlFloatArray>(innerNodes,  flatIdx.xmin2),
                                 eavlIndexable<eavlFloatArray>(innerNodes,  flatIdx.ymin2),
                                 eavlIndexable<eavlFloatArray>(innerNodes,  flatIdx.zmin2),
                                 eavlIndexable<eavlFloatArray>(innerNodes,  flatIdx.xmax2),
                                 eavlIndexable<eavlFloatArray>(innerNodes,  flatIdx.ymax2),
                                 eavlIndexable<eavlFloatArray>(innerNodes,  flatIdx.zmax2),
                                 eavlIndexable<eavlFloatArray>(innerNodes,  flatIdx.lChild),
                                 eavlIndexable<eavlFloatArray>(innerNodes,  flatIdx.rChild)),
                      InnerToFlatFunctor(xmins, ymins, zmins,
                                         xmaxs, ymaxs, zmaxs, 
                                         numPrimitives), numPrimitives - 1),
                      "write");
    eavlExecutor::Go();

    eavlArrayIndexer numPrim;
    numPrim.mul = 2;
    eavlArrayIndexer id;
    id.mul = 2;
    id.add = 1;
    eavlExecutor::AddOperation(
        new_eavlMapOp(eavlOpArgs(bvh->primId),
                      eavlOpArgs(eavlIndexable<eavlIntArray>(leafNodes,  numPrim),
                                 eavlIndexable<eavlIntArray>(leafNodes,  id)),
                      LeafToFlatFunctor()),
                      "write");
    eavlExecutor::Go();
}

float * MortonBVHBuilder::getInnerNodes(int &_size)
{ 
    if(!convertedToAoS)
    {
        flatten();
        convertedToAoS = true;
    }

    int size = (numPrimitives -1) * 16;
    float * array =  new float[size];
    memcpy((void*)array, innerNodes->GetHostArray(), sizeof(float) * size);
    _size = size;
    return array; 
}
int * MortonBVHBuilder::getLeafNodes(int &_size)
{ 
    if(!convertedToAoS)
    {
        flatten();
        convertedToAoS = true;
    }
    int size = numPrimitives * 2;
    int * array =  new int[size];
    memcpy((void*)array, leafNodes->GetHostArray(), sizeof(int) * size);
    _size = size;
    return array; 
}

eavlFloatArray * MortonBVHBuilder::getInnerNodes()
{
    if(!convertedToAoS)
    {
        flatten();
        convertedToAoS = true;
    }
    wasEavlArrayGiven = true;
    return innerNodes;
}

eavlIntArray * MortonBVHBuilder::getLeafNodes()
{
    if(!convertedToAoS)
    {
        flatten();
        convertedToAoS = true;
    }
    wasEavlArrayGiven = true;
    return leafNodes;
}
