#include "eavlTriangleIntersector.h"


#define TOLERANCE   0.00001
#define BARY_TOLE   0.0001f
#define EPSILON     0.01f
#define INFINITE    1000000
#define END_FLAG    -1000000000

/* Triangle textures */
texture<float4> tri_bvh_in_tref;            /* BVH inner nodes */
texture<float4> tri_verts_tref;             /* vert+ scalar data */
texture<float>  tri_bvh_lf_tref;            /* BVH leaf nodes */
texture<float>  tri_norms_tref;

EAVL_HOSTDEVICE int getIntersectionTri(const eavlVector3 rayDir, const eavlVector3 rayOrigin, bool occlusion, const eavlConstTexArray<float4> *bvh,
                                       const eavlConstTexArray<float> *tri_bvh_lf_raw, const eavlConstTexArray<float4> *verts,const float &maxDistance, float &distance)
{


    float minDistance = maxDistance;
    int   minIndex    = -1;
    
    float dirx = rayDir.x;
    float diry = rayDir.y;
    float dirz = rayDir.z;

    float invDirx = rcp_safe(dirx);
    float invDiry = rcp_safe(diry);
    float invDirz = rcp_safe(dirz);
    int currentNode;
  
    int todo[64]; //num of nodes to process
    int stackptr = 0;
    int barrier = (int)END_FLAG;
    currentNode = 0;

    todo[stackptr] = barrier;

    float ox = rayOrigin.x;
    float oy = rayOrigin.y;
    float oz = rayOrigin.z;
    float odirx = ox * invDirx;
    float odiry = oy * invDiry;
    float odirz = oz * invDirz;

    while(currentNode != END_FLAG) 
    {

        if(currentNode>-1)
        {

            float4 n1 = bvh->getValue(tri_bvh_in_tref, currentNode  ); //(txmin0, tymin0, tzmin0, txmax0)
            float4 n2 = bvh->getValue(tri_bvh_in_tref, currentNode+1); //(tymax0, tzmax0, txmin1, tymin1)
            float4 n3 = bvh->getValue(tri_bvh_in_tref, currentNode+2); //(tzmin1, txmax1, tymax1, tzmax1)
            
            float txmin0 = n1.x * invDirx - odirx;       
            float tymin0 = n1.y * invDiry - odiry;         
            float tzmin0 = n1.z * invDirz - odirz;
            float txmax0 = n1.w * invDirx - odirx;
            float tymax0 = n2.x * invDiry - odiry;
            float tzmax0 = n2.y * invDirz - odirz;
           
            float tmin0 = max(max(max(min(tymin0,tymax0),min(txmin0,txmax0)),min(tzmin0,tzmax0)),0.f);
            float tmax0 = min(min(min(max(tymin0,tymax0),max(txmin0,txmax0)),max(tzmin0,tzmax0)), minDistance);
            
            bool traverseChild0 = (tmax0 >= tmin0);

            float txmin1 = n2.z * invDirx - odirx;       
            float tymin1 = n2.w * invDiry - odiry;
            float tzmin1 = n3.x * invDirz - odirz;
            float txmax1 = n3.y * invDirx - odirx;
            float tymax1 = n3.z * invDiry-  odiry;
            float tzmax1 = n3.w * invDirz - odirz;
            float tmin1 = max(max(max(min(tymin1,tymax1),min(txmin1,txmax1)),min(tzmin1,tzmax1)),0.f);
            float tmax1 = min(min(min(max(tymin1,tymax1),max(txmin1,txmax1)),max(tzmin1,tzmax1)), minDistance);
            
            bool traverseChild1 = (tmax1 >= tmin1);

	        if(!traverseChild0 && !traverseChild1)
	        {

	            currentNode = todo[stackptr]; //go back put the stack
	            stackptr--;
	        }
	        else
	        {
	            float4 n4 = bvh->getValue(tri_bvh_in_tref, currentNode+3); //(leftChild, rightChild, pad,pad)
	            int leftChild = (int)n4.x;
	            int rightChild = (int)n4.y;

	            currentNode = (traverseChild0) ? leftChild : rightChild;
	            if(traverseChild1 && traverseChild0)
	            {
	                if(tmin0 > tmin1)
	                {
	                    currentNode = rightChild;
	                    stackptr++;
	                    todo[stackptr] = leftChild;
	                }
	                else
	                {   
	                    stackptr++;
	                    todo[stackptr] = rightChild;
	                }
	            }
	        }
    	}//if inner node
        
        if(currentNode < 0 && currentNode != barrier)//check register usage
        {
            currentNode = -currentNode - 1; //swap the neg address 
            int numTri = (int)tri_bvh_lf_raw->getValue(tri_bvh_lf_tref,currentNode)+1;

            for(int i = 1; i < numTri; i++)
            {        
                    int triIndex = (int)tri_bvh_lf_raw->getValue(tri_bvh_lf_tref,currentNode+i);
                   
                    float4 a4 = verts->getValue(tri_verts_tref, triIndex*3);
                    float4 b4 = verts->getValue(tri_verts_tref, triIndex*3+1);
                    float4 c4 = verts->getValue(tri_verts_tref, triIndex*3+2);
                    eavlVector3 e1( a4.w - a4.x , b4.x - a4.y, b4.y - a4.z ); 
                    eavlVector3 e2( b4.z - a4.x , b4.w - a4.y, c4.x - a4.z );


                    eavlVector3 p;
                    p.x = diry * e2.z - dirz * e2.y;
                    p.y = dirz * e2.x - dirx * e2.z;
                    p.z = dirx * e2.y - diry * e2.x;
                    float dot = e1.x * p.x + e1.y * p.y + e1.z * p.z;
                    if(dot != 0.f)
                    {
                        dot = 1.f/dot;
                        eavlVector3 t;
                        t.x = ox - a4.x;
                        t.y = oy - a4.y;
                        t.z = oz - a4.z;

                        float u = (t.x* p.x + t.y * p.y + t.z * p.z) * dot;
                        if(u >= (0.f - EPSILON) && u <= (1.f + EPSILON))
                        {
                            eavlVector3 q; // = t % e1;
                            q.x = t.y * e1.z - t.z * e1.y;
                            q.y = t.z * e1.x - t.x * e1.z;
                            q.z = t.x * e1.y - t.y * e1.x;
                            float v = (dirx * q.x + diry * q.y + dirz * q.z) * dot;
                            if(v >= (0.f - EPSILON) && v <= (1.f + EPSILON))
                            {
                                float dist = (e2.x * q.x + e2.y * q.y + e2.z * q.z) * dot;
                                if((dist > EPSILON && dist < minDistance) && !(u + v > 1) )
                                {
                                    minDistance = dist;
                                    minIndex = triIndex;
                                    if(occlusion) return minIndex;//or set todo to -1
                                }//if
                            }//if
                        }//if

                    }//if parallel
                   
            }//each shape
            currentNode = todo[stackptr];
            stackptr--;
        }//if inner node

    }//while
 distance = minDistance;
 return minIndex;
}

template <bool occlusion> 
struct RayTriangleIntersectionFunctor{

    const eavlConstTexArray<float4> *verts;
    const eavlConstTexArray<float4> *bvh;
    const eavlConstTexArray<float>  *bvh_inner;

    RayTriangleIntersectionFunctor(const eavlConstTexArray<float4> *_verts, 
    							   const eavlConstTexArray<float4> *theBvh,
                       			   const eavlConstTexArray<float> *_bvh_inner)
        :verts(_verts),
         bvh(theBvh),
         bvh_inner(_bvh_inner)
    {}                                                 
    EAVL_HOSTDEVICE tuple<int,float,int> operator()( tuple<float,float,float,float,float,float,int, int, float> rayTuple)
    {
       
        
        int hitIdx = get<6>(rayTuple);
        if(hitIdx == -1) return tuple<int,float,int>(-1, INFINITE, -1);

        int   minHit = -1;    
        float minDistance = get<8>(rayTuple);
	    
	    float dirx = get<0>(rayTuple);
	    float diry = get<1>(rayTuple);
	    float dirz = get<2>(rayTuple);

	    float invDirx = rcp_safe(dirx);
	    float invDiry = rcp_safe(diry);
	    float invDirz = rcp_safe(dirz);
	    int currentNode;
	  
	    int todo[64]; //num of nodes to process
	    int stackptr = 0;
	    int barrier = (int)END_FLAG;
	    currentNode = 0;

	    todo[stackptr] = barrier;

	    float ox = get<3>(rayTuple);
	    float oy = get<4>(rayTuple);
	    float oz = get<5>(rayTuple);
	    float odirx = ox * invDirx;
	    float odiry = oy * invDiry;
	    float odirz = oz * invDirz;

	    while(currentNode != END_FLAG) 
	    {

	        if(currentNode>-1)
	        {

	            float4 n1 = bvh->getValue(tri_bvh_in_tref, currentNode  ); //(txmin0, tymin0, tzmin0, txmax0)
	            float4 n2 = bvh->getValue(tri_bvh_in_tref, currentNode+1); //(tymax0, tzmax0, txmin1, tymin1)
	            float4 n3 = bvh->getValue(tri_bvh_in_tref, currentNode+2); //(tzmin1, txmax1, tymax1, tzmax1)
	            
	            float txmin0 = n1.x * invDirx - odirx;       
	            float tymin0 = n1.y * invDiry - odiry;         
	            float tzmin0 = n1.z * invDirz - odirz;
	            float txmax0 = n1.w * invDirx - odirx;
	            float tymax0 = n2.x * invDiry - odiry;
	            float tzmax0 = n2.y * invDirz - odirz;
	           
	            float tmin0 = max(max(max(min(tymin0,tymax0),min(txmin0,txmax0)),min(tzmin0,tzmax0)),0.f);
	            float tmax0 = min(min(min(max(tymin0,tymax0),max(txmin0,txmax0)),max(tzmin0,tzmax0)), minDistance);
	            
	            bool traverseChild0 = (tmax0 >= tmin0);

	            float txmin1 = n2.z * invDirx - odirx;       
	            float tymin1 = n2.w * invDiry - odiry;
	            float tzmin1 = n3.x * invDirz - odirz;
	            float txmax1 = n3.y * invDirx - odirx;
	            float tymax1 = n3.z * invDiry-  odiry;
	            float tzmax1 = n3.w * invDirz - odirz;
	            float tmin1 = max(max(max(min(tymin1,tymax1),min(txmin1,txmax1)),min(tzmin1,tzmax1)),0.f);
	            float tmax1 = min(min(min(max(tymin1,tymax1),max(txmin1,txmax1)),max(tzmin1,tzmax1)), minDistance);
	            
	            bool traverseChild1 = (tmax1 >= tmin1);

		        if(!traverseChild0 && !traverseChild1)
		        {

		            currentNode = todo[stackptr]; //go back put the stack
		            stackptr--;
		        }
		        else
		        {
		            float4 n4 = bvh->getValue(tri_bvh_in_tref, currentNode+3); //(leftChild, rightChild, pad,pad)
		            int leftChild = (int)n4.x;
		            int rightChild = (int)n4.y;

		            currentNode = (traverseChild0) ? leftChild : rightChild;
		            if(traverseChild1 && traverseChild0)
		            {
		                if(tmin0 > tmin1)
		                {
		                    currentNode = rightChild;
		                    stackptr++;
		                    todo[stackptr] = leftChild;
		                }
		                else
		                {   
		                    stackptr++;
		                    todo[stackptr] = rightChild;
		                }
		            }
		        }
	    	}//if inner node
	        
	        if(currentNode < 0 && currentNode != barrier)//check register usage
	        {
	            currentNode = -currentNode - 1; //swap the neg address 
	            int numTri = (int)tri_bvh_lf_raw->getValue(tri_bvh_lf_tref,currentNode)+1;

	            for(int i = 1; i < numTri; i++)
	            {        
	                    int triIndex = (int)tri_bvh_lf_raw->getValue(tri_bvh_lf_tref,currentNode+i);
	                   
	                    float4 a4 = verts->getValue(tri_verts_tref, triIndex*3);
	                    float4 b4 = verts->getValue(tri_verts_tref, triIndex*3+1);
	                    float4 c4 = verts->getValue(tri_verts_tref, triIndex*3+2);
	                    eavlVector3 e1( a4.w - a4.x , b4.x - a4.y, b4.y - a4.z ); 
	                    eavlVector3 e2( b4.z - a4.x , b4.w - a4.y, c4.x - a4.z );


	                    eavlVector3 p;
	                    p.x = diry * e2.z - dirz * e2.y;
	                    p.y = dirz * e2.x - dirx * e2.z;
	                    p.z = dirx * e2.y - diry * e2.x;
	                    float dot = e1.x * p.x + e1.y * p.y + e1.z * p.z;
	                    if(dot != 0.f)
	                    {
	                        dot = 1.f/dot;
	                        eavlVector3 t;
	                        t.x = ox - a4.x;
	                        t.y = oy - a4.y;
	                        t.z = oz - a4.z;

	                        float u = (t.x* p.x + t.y * p.y + t.z * p.z) * dot;
	                        if(u >= (0.f - EPSILON) && u <= (1.f + EPSILON))
	                        {
	                            eavlVector3 q; // = t % e1;
	                            q.x = t.y * e1.z - t.z * e1.y;
	                            q.y = t.z * e1.x - t.x * e1.z;
	                            q.z = t.x * e1.y - t.y * e1.x;
	                            float v = (dirx * q.x + diry * q.y + dirz * q.z) * dot;
	                            if(v >= (0.f - EPSILON) && v <= (1.f + EPSILON))
	                            {
	                                float dist = (e2.x * q.x + e2.y * q.y + e2.z * q.z) * dot;
	                                if((dist > EPSILON && dist < minDistance) && !(u + v > 1) )
	                                {
	                                    minDistance = dist;
	                                    minHit = triIndex;
	                                    if(occlusion) return minIndex;//or set todo to -1
	                                }//if
	                            }//if
	                        }//if

	                    }//if parallel    
	            }//each shape
	            currentNode = todo[stackptr];
	            stackptr--;
	        }//if inner node
	    }//while
    if(minHit!=-1) return tuple<int,float,int>(minHit, minDistance, primitiveType);
    else           return tuple<int,float,int>(hitIdx, INFINITE, get<7>(rayTuple));
    }
};