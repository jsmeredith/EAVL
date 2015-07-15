#ifndef EAVL_RAY_DEFINES
#define EAVL_RAY_DEFINES

#define TOLERANCE   0.00001
#define BARY_TOLE   0.0001f
#define EPSILON     0.01f
#define INFINITE    1000000
#define END_FLAG    -1000000000

#ifndef __CUDACC__
template<class T> class texture {};

#ifndef HAVE_CUDA
struct float4
{
    float x,y,z,w;
};

struct int4
{
	int x,y,z,w;
};

#endif
#endif



#endif