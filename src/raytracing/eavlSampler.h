#ifndef EAVL_SAMPLER_H
#define EAVL_SAMPLER_H
#include "eavlVector3.h"
#include "eavlMatrix4x4.h"
/*

    Sampling class of quasi-random sequences.
    Important note: sample numbers should be sequential
    or the resulting samples will be a good distribution.

*/
class eavlSampler
{
  public:
    enum Sampler_t { HALTON, HAMMERSLY };

  	static EAVL_HOSTDEVICE void jenkinsMix(unsigned int & a, unsigned int & b, unsigned int & c)
	{
	    a -= b; a -= c; a ^= (c>>13);
	    b -= c; b -= a; b ^= (a<<8);
	    c -= a; c -= b; c ^= (b>>13);
	    a -= b; a -= c; a ^= (c>>12);
	    b -= c; b -= a; b ^= (a<<16);
	    c -= a; c -= b; c ^= (b>>5);
	    a -= b; a -= c; a ^= (c>>3);
	    b -= c; b -= a; b ^= (a<<10);
	    c -= a; c -= b; c ^= (b>>15);   
	}
	
  	template<int base>
	static void EAVL_HOSTDEVICE halton2D(const int &sampleNum, float *coord)
	{
		//generate base2 halton
		float  x = 0.0f;
        float  xadd = 1.0f;
        unsigned int b2 = 1 + sampleNum;
        while (b2 != 0)
        {
            xadd *= 0.5f;
            if ((b2 & 1) != 0)
                x += xadd;
            b2 >>= 1;
        }

        float  y = 0.0f;
        float  yadd = 1.0f;
        int bn = 1 + sampleNum;
        while (bn != 0)
        {
            yadd *= 1.0f / (float) base;
            y += (float)(bn % base) * yadd;
            bn /= base;
        }

        coord[0] = x;
        coord[1] = y;
	}

    template<int base>
    static void EAVL_HOSTDEVICE hammersly2D(const int &sampleNum, float *coord)
    {
        //generate base2 hammersly
        float  y = 0.0f;
        float  yadd = 1.0f;
        unsigned int b2 = 1 + sampleNum;
        float totSamples = sampleNum + 1000.f;
        while (b2 != 0)
        {
            yadd *= 0.5f;
            if ((b2 & 1) != 0)
                y += yadd;
            b2 >>= 1;
        }

        float x  = (float)sampleNum / totSamples; // TODO: need a way to get the number of samples for this to work

        coord[0] = x;
        coord[1] = y;
        
    }

    template<Sampler_t stype>
	static eavlVector3 EAVL_HOSTDEVICE hemisphere(const int &sampleNum,
                                                  const int &seed,
												  const eavlVector3 &normal)
	{
		eavlVector3 absNormal;
        absNormal.x = abs(normal.x);
        absNormal.y = abs(normal.y);
        absNormal.z = abs(normal.z);
        
        float maxN = max(max(absNormal.x,absNormal.y),absNormal.z);
        eavlVector3 perp = eavlVector3(normal.y,-normal.x,0.f);
        if(maxN == absNormal.z)  
        {
            perp.x = 0.f;
            perp.y = normal.z;
            perp.z = -normal.y;
        }
        else if (maxN == absNormal.x)
        {
            perp.x = -normal.z;
            perp.y = 0.f;
            perp.z = normal.x;
        }
        perp.normalize(); 

        eavlVector3 biperp = normal % perp;
        unsigned int hashA = 6371625 + seed;
        unsigned int hashB = 0x9e3779b9u;
        unsigned int hashC = 0x9e3779b9u;
        jenkinsMix(hashA, hashB, hashC);
        jenkinsMix(hashA, hashB, hashC);

        float angle = 2.f * PI * (float)hashC * exp2(-32.f);

        eavlVector3 t0 = perp * cosf(angle) + biperp * sinf(angle);
        eavlVector3 t1 = perp * -sinf(angle) + biperp * cosf(angle);

        float xy[2];

        if(stype == HALTON)
        {
            eavlSampler::template halton2D<3>(sampleNum +seed, xy);
        }
        else if(stype == HAMMERSLY)
        {
            eavlSampler::template hammersly2D<3>(sampleNum+seed, xy);
        }
        //cout<<xy[0]<<" "<<xy[1]<<endl;

        float angle2 = 2.0f * PI * xy[1];
        float r = sqrtf(xy[0]);
        xy[0] = r * cosf(angle2);
        xy[1] = r * sinf(angle2);
        float z = sqrtf(1.0f - xy[0] * xy[0] - xy[1] * xy[1]);
        eavlVector3 dir = eavlVector3( t0*xy[0]  +  t1*xy[1] +  normal*z);
        dir.normalize();
        return dir;
    }

    template<Sampler_t stype>
    static eavlVector3 EAVL_HOSTDEVICE importanceSampleHemi(const int &sampleNum, 
                                                 const eavlVector3 &normal,
                                                 const float &shine,
                                                 float &weight, int seed)
    {

        //int seed = 100; //TODO: add seed, maybe fframenumber or texture lookup

        eavlVector3 absNormal;
        absNormal.x = abs(normal.x);
        absNormal.y = abs(normal.y);
        absNormal.z = abs(normal.z);
        
        float maxN = max(max(absNormal.x,absNormal.y),absNormal.z);
        eavlVector3 perp = eavlVector3(normal.y,-normal.x,0.f);
        if(maxN == absNormal.z)  
        {
            perp.x = 0.f;
            perp.y = normal.z;
            perp.z = -normal.y;
        }
        else if (maxN == absNormal.x)
        {
            perp.x = -normal.z;
            perp.y = 0.f;
            perp.z = normal.x;
        }
        perp.normalize(); 

        eavlVector3 biperp = normal % perp;
        unsigned int hashA = 6371625 + seed;
        unsigned int hashB = 0x9e3779b9u;
        unsigned int hashC = 0x9e3779b9u;
        jenkinsMix(hashA, hashB, hashC);
        jenkinsMix(hashA, hashB, hashC);

        float angle = 2.f * PI * (float)hashC * exp2(-32.f);

        eavlVector3 t0 = perp * cosf(angle) + biperp * sinf(angle);
        eavlVector3 t1 = perp * -sinf(angle) + biperp * cosf(angle);

        float xy[2];

        if(stype == HALTON)
        {
            eavlSampler::template halton2D<3>(sampleNum, xy);
        }
        else if(stype == HAMMERSLY)
        {
            eavlSampler::template hammersly2D<3>(sampleNum, xy);
        }
        else
        {
            printf("NO SAMPLE\n");
        }

        //printf("X %f Y %f\n", xy[0], xy[1]);
        //get random spherical coords 
        
        float theta = acos(pow(xy[0], 1.0f / (shine + 1.0f)));
        float phi = 2.0f * M_PI * xy[1];
        //printf("theta %f phi %f\n", theta, phi);
        //get the cartesian coords
        eavlVector3 dir;
        float sintheta = sin(theta); 
        float costheta = cos(theta);  //there could be a better way to calc this
        dir.x = cos(phi) * sintheta;
        dir.y = sin(phi) * sintheta;
        dir.z = costheta;
        // dir.x = 0;
        // dir.y = 0;
        // dir.z = 1;
        //cout<<"perp "<<perp<<" biperp "<<biperp<<" normal "<<normal<<endl;
        eavlMatrix4x4 trans(t0.x, t0.y, t0.z, 0.0f,
                            t1.x, t1.y, t1.z, 0.0f,
                            normal.x, normal.y, normal.z, 0.0f,
                            0.0f, 0.0f, 0.0f, 0.0f);
        trans.Transpose();
        //cout<<"Before "<<dir<<endl;
        dir = trans * dir;

        
        dir.normalize();
        
        //cout<<"Data  "<<dir<<endl<<" "<<trans<<" ";

        float pdf = (shine + 1) / (2.0 * M_PI);
        pdf *=  pow(costheta, shine);
        if(costheta < 0 ) printf("No cos < 0 seed %d\n",seed);
        weight = 1.0 / pdf;
        if(weight < 0) printf("Can't have a negative weight seed %d\n",seed);
        //printf("Weight %f Angle %f shine %f\n",weight, theta,shine);
        
        return dir;

    }

    template<Sampler_t stype>
    static eavlVector3 EAVL_HOSTDEVICE sphereSample(const int &sampleNum, 
                                                    const float radius)
    {

        float xy[2];

        if(stype == HALTON)
        {
            eavlSampler::template halton2D<3>(sampleNum, xy);
        }
        else if(stype == HAMMERSLY)
        {
            eavlSampler::template hammersly2D<3>(sampleNum, xy);
        }
        else
        {
            printf("NO SAMPLE\n");
        }

        //map to [-1, 1] and [0, 2* pi]
        xy[0] = xy[0] * 2.f - 1.f;
        xy[1] = xy[1] * 2.f * M_PI; 
        
        float st = sqrt(1.f - xy[0]);

        eavlVector3 sample;
        sample.x = st * cos(xy[1]);
        sample.y = st * sin(xy[0]);
        sample.z = xy[0];

        return sample * radius;

    }

};

#endif