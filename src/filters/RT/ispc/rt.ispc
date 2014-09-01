
#define EPSILON     0.01f
#define INFINITE    1000000
#define END_FLAG    -1000000000

//typedef float float3[3];
//typedef float float4[4];

struct float3{
	float x;
	float y;
	float z;
};

struct float4{
	float x;
	float y; 
	float z;
	float w;
};

float srcp(float f){ return 1.0f/f;}
float rcp_safe(float f) { return srcp((abs(f) < 1e-8f) ? 1e-8f : f); }

void intersectTri(uniform float rdx[], uniform float rdy[], uniform float rdz[],
						 uniform float rox[], uniform float roy[], uniform float roz[],
						 uniform float4 bvh_in[], uniform float bvh_lf[],
						 uniform float minIdxs[], uniform float minDistances[],
						 uniform float4 verts[], uniform int first, uniform int last)
{

	foreach(idx = first ... last)
	{
	 	float minDistance = INFINITE;
	    int   minIndex    = -1;
	    
	    float dirx = rdx[idx];//rayDir.x;
	    float diry = rdy[idx];//rayDir.y;
	    float dirz = rdz[idx];//rayDir.z;

	    float invDirx = rcp_safe(dirx);
	    float invDiry = rcp_safe(diry);
	    float invDirz = rcp_safe(dirz);
	    int currentNode;
	  
	    int todo[64]; //num of nodes to process
	    int stackptr = 0;
	    int barrier = (int)END_FLAG;
	    currentNode = 0;

	    todo[stackptr] = barrier;

	    float ox = rox[idx];//rayOrigin.x;
	    float oy = roy[idx];//rayOrigin.y;
	    float oz = roz[idx];//rayOrigin.z;
	    float odirx = ox * invDirx;
	    float odiry = oy * invDiry;
	    float odirz = oz * invDirz;

	    while(currentNode != END_FLAG) {

	        if(currentNode > -1)
	        {

	            float4 n1 = bvh_in[currentNode   ]; //(txmin0, tymin0, tzmin0, txmax0)
	            float4 n2 = bvh_in[currentNode + 1]; //(tymax0, tzmax0, txmin1, tymin1)
	            float4 n3 = bvh_in[currentNode + 2]; //(tzmin1, txmax1, tymax1, tzmax1)
	            
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
	            float4 n4 = bvh_in[currentNode + 3]; //(leftChild, rightChild, pad,pad)
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
	        }
	        
	        if(currentNode < 0 && currentNode != barrier)//check register usage
	        {
	            

	            currentNode = -currentNode; //swap the neg address 
	            int numTri = (int)bvh_lf[currentNode]+1;

	            for(int i = 1; i < numTri; i++)
	            {        
	                    int triIndex = (int)bvh_lf[currentNode + i];
	                   
	                    float4 a4 = verts[triIndex * 3    ];
	                    float4 b4 = verts[triIndex * 3 + 1];
	                    float4 c4 = verts[triIndex * 3 + 2];
	                    float3 e1;
	                   	e1.x = a4.w - a4.x;
	                   	e1.y = b4.x - a4.y;
	                   	e1.z = b4.y - a4.z;
	                    float3 e2;
	                    e2.x = b4.z - a4.x;
	                    e2.y = b4.w - a4.y;
	                    e2.z = c4.x - a4.z;

	                    float3 p;
	                    p.x = diry * e2.z - dirz * e2.y;
	                    p.y = dirz * e2.x - dirx * e2.z;
	                    p.z = dirx * e2.y - diry * e2.x;
	                    float dot = e1.x * p.x + e1.y * p.y + e1.z * p.z;
	                    if(dot != 0.f)
	                    {
	                        dot = 1.f/dot;
	                        float3 t;
	                        t.x = ox - a4.x;
	                        t.y = oy - a4.y;
	                        t.z = oz - a4.z;

	                        float u = (t.x* p.x + t.y * p.y + t.z * p.z) * dot;
	                        if(u >= (0.f - EPSILON) && u <= (1.f + EPSILON))
	                        {
	                            float3 q; // = t % e1;
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
	                                    //if(occlusion) return minIndex;//or set todo to -1
	                                }
	                            }
	                        }

	                    }
	                   
	            }
	            currentNode = todo[stackptr];
	            stackptr--;
	        }

	    }
	 	minDistances[idx] = minDistance;
	 	minIdxs[idx] = minIndex;
	}

}

export void intersectTriHelper(uniform float rdx[], uniform float rdy[], uniform float rdz[],
						 	   uniform float rox[], uniform float roy[], uniform float roz[],
						 	   uniform float bvh_in[], uniform float bvh_lf[],
						       uniform float minIdxs[], uniform float minDistances[],
						       uniform float verts[], uniform int first,  uniform int last)
{
	intersectTri(rdx, rdy, rdz,
				 rox, roy, roz,
			     (float4 *)bvh_in, bvh_lf,
				 minIdxs, minDistances,
			     (float4 *) verts, first, last);
}