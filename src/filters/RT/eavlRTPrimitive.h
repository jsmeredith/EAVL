#ifndef EAVL_RT_PRIMITIVE_H
#define EAVL_RT_PRIMITIVE_H

struct BBox {
	eavlVector3 min, max, extent;
	int count;
	EAVL_HOSTDEVICE BBox() 
	{
		
	 }
	EAVL_HOSTDEVICE BBox(const eavlVector3& min, const eavlVector3& max): min(min), max(max)
	{ 
		extent = max - min; 
	}
	EAVL_HOSTDEVICE BBox(const eavlVector3& p): min(p), max(p)
	{ 
		extent = max - min; 
	}

	EAVL_HOSTDEVICE void expandToInclude(const eavlVector3& p)
	{
		min.x = min.x > p.x ? p.x : min.x;
		min.y = min.y > p.y ? p.y : min.y;
		min.z = min.z > p.z ? p.z : min.z;
		max.x = max.x < p.x ? p.x : max.x;
		max.y = max.y < p.y ? p.y : max.y;
		max.z = max.z < p.z ? p.z : max.z;
		extent = max - min;
	}

	EAVL_HOSTDEVICE void clear()
	{
		min.x=1000000;
		min.y=1000000;
		min.z=1000000;
		max.x=-1000000;
		max.y=-1000000;
		max.z=-1000000; 
		extent.x=0;
		extent.y=0;
		extent.z=0;
		count=0;


	}

	EAVL_HOSTDEVICE void expandToInclude(const BBox& b)
	{
	 	expandToInclude(b.min);
	 	expandToInclude(b.max);
	 	extent = max - min;
	 	count++;
	}
    EAVL_HOSTDEVICE int maxDimension() const 
    {
		int result = 0;
		//if(extent.y > extent.x) result = 1;
		//if(extent.z > extent.y) result = 2;
		return result;
	}
 	EAVL_HOSTDEVICE float surfaceArea() const 
 	{
 		eavlVector3 extent = max - min; 
		return 2.f*( extent.x*extent.z + extent.x*extent.y + extent.y*extent.z );
	}
};

struct RTMaterial
{
	eavlVector3 ka,kd, ks; 	//specular and diffuse components
	float 		shiny; 		//specular exponent
	float 		rs;			//percentage of refected specular light
	EAVL_HOSTDEVICE RTMaterial()
	{
		ka.x=.7f;
		ka.y=.7f;
		ka.z=.7f;

		kd.x=.7f;			// I just made these values up
		kd.y=.7f;
		kd.z=.7f;

		ks.x=.7f;
		ks.y=.7f;
		ks.z=.7f;

		shiny=10.f;
		rs=.2f;
	}

	EAVL_HOSTDEVICE RTMaterial(float r, float g, float b)
	{
		ka.x=r;
		ka.y=g;
		ka.z=b;

		kd.x=r;			// I just made these values up
		kd.y=g;
		kd.z=b;

		ks.x=r;
		ks.y=g;
		ks.z=b;

		shiny=5.f;
		rs=.2f;
	}

	EAVL_HOSTDEVICE RTMaterial(eavlVector3 _ka, eavlVector3 _kd, eavlVector3 _ks, float _shiny, float _rs)
	{
		ka=_ka;
		kd=_kd;
		ks=_ks;
		shiny=_shiny;
		rs=_rs;
	}
};


class RTPrimitive
{
	protected:
		BBox bbox;
		int matIdx; 	//index 
		//maybe centroid?
	public:
		EAVL_HOSTDEVICE virtual float 	getSurfaceArea() const {return 0.f;};
		EAVL_HOSTDEVICE const 	BBox  	getBBox(){ return bbox; }
		EAVL_HOSTDEVICE int   			getMatIndex(){return matIdx;}
		EAVL_HOSTDEVICE void	    	setMatIndex(const int &idx) {matIdx=idx;}
};

class RTTriangle : public RTPrimitive
{

	public:
		eavlVector3 verts[3];
		eavlVector3 norms[3];
		float 		scalars[3];
		EAVL_HOSTDEVICE RTTriangle(const eavlVector3 &_v0, const eavlVector3 &_v1, const eavlVector3 &_v2,
								   const float &_scalarV0, const float &_scalarV1, const float &_scalarV2, const int &_matIdx=0)
		{
			matIdx=_matIdx; 
			bbox.expandToInclude(_v0);
			bbox.expandToInclude(_v1);
			bbox.expandToInclude(_v2);
			verts[0]=_v0;
			verts[1]=_v1;
			verts[2]=_v2;
			eavlVector3 normal=(verts[1]-verts[0])%(verts[2]-verts[0]);
			normal.normalize();
			norms[0]= normal;
			norms[1]= normal;
			norms[2]= normal;
			scalars[0]=_scalarV0;
			scalars[1]=_scalarV1;
			scalars[2]=_scalarV2;


		}
		EAVL_HOSTDEVICE RTTriangle(const eavlVector3 &_v0, const eavlVector3 &_v1, const eavlVector3 &_v2,
								   const eavlVector3 &_n0, const eavlVector3 &_n1, const eavlVector3 &_n2,
								   const float &_scalarV0, const float &_scalarV1, const float &_scalarV2, const int &_matIdx=0)
		{
			matIdx=_matIdx; 
			bbox.expandToInclude(_v0);
			bbox.expandToInclude(_v1);
			bbox.expandToInclude(_v2);
			verts[0]=_v0;
			verts[1]=_v1;
			verts[2]=_v2;
			norms[0]=_n0;
			norms[1]=_n1;
			norms[2]=_n2;
			scalars[0]=_scalarV0;
			scalars[1]=_scalarV1;
			scalars[2]=_scalarV2;

		}
		/*
		EAVL_HOSTDEVICE RTTriangle(const float * triPtr, const int &_matIdx=0)
		{
			matIdx=_matIdx;
			verts[0].x=triPtr[0];
			verts[0].y=triPtr[1];
			verts[0].z=triPtr[2];

			verts[1].x=triPtr[3];
			verts[1].y=triPtr[4];
			verts[1].z=triPtr[5];

			verts[2].x=triPtr[6];
			verts[2].y=triPtr[7];
			verts[2].z=triPtr[8];

			eavlVector3 normal=verts[0]%verts[1];
			normal.normalize();
			norms[0]= normal;
			norms[1]= normal;
			norms[2]= normal;

			bbox.expandToInclude(verts[0]);
			bbox.expandToInclude(verts[1]);
			bbox.expandToInclude(verts[2]);
		}
		
		EAVL_HOSTDEVICE RTTriangle(const float * triPtr, const float * normPtr, const int &_matIdx=0)
		{
			matIdx=_matIdx;
			verts[0].x=triPtr[0];
			verts[0].y=triPtr[1];
			verts[0].z=triPtr[2];

			verts[1].x=triPtr[3];
			verts[1].y=triPtr[4];
			verts[1].z=triPtr[5];

			verts[2].x=triPtr[6];
			verts[2].y=triPtr[7];
			verts[2].z=triPtr[8];

			norms[0].x=normPtr[0];
			norms[0].y=normPtr[1];
			norms[0].z=normPtr[2];

			norms[1].x=normPtr[3];
			norms[1].y=normPtr[4];
			norms[1].z=normPtr[5];

			norms[2].x=normPtr[6];
			norms[2].y=normPtr[7];
			norms[2].z=normPtr[8];
			
			bbox.expandToInclude(verts[0]);
			bbox.expandToInclude(verts[1]);
			bbox.expandToInclude(verts[2]);
		}
		*/
		EAVL_HOSTDEVICE float getSurfaceArea() const 
		{
			eavlVector3 norm=(verts[2]-verts[0])%(verts[1]-verts[0]);
			float mag= sqrt(norm.x*norm.x+norm.y*norm.y+norm.z*norm.z);
			return .5f*mag;
		}

};




class RTSphere : public RTPrimitive
{
		
	public:
		float data[4]; //center + radius
		float scalar;
		EAVL_HOSTDEVICE RTSphere(const float &_radius, const eavlVector3 &_center,const float _scalar, const int &_matIdx )
		{
			matIdx = _matIdx;
			eavlVector3 temp(0,0,0);
			temp.x = _radius;
			temp.y = 0;
			temp.z = 0;
			bbox.expandToInclude(_center+temp);
			bbox.expandToInclude(_center-temp);
			temp.x = 0;
			temp.y = _radius;
			temp.z = 0;
			bbox.expandToInclude(_center+temp);
			bbox.expandToInclude(_center-temp);
			temp.x = 0;
			temp.y = 0;
			temp.z = _radius;
			bbox.expandToInclude(_center+temp);
			bbox.expandToInclude(_center-temp);
			data[0] = _center.x;
			data[1] = _center.y;
			data[2] = _center.z;
			data[3] = _radius;
			scalar  = _scalar;
		}

		EAVL_HOSTDEVICE int 		  getRadius() 		const  { return data[3]; }
		EAVL_HOSTDEVICE eavlVector3   getCenter()      	const  { return eavlVector3(data[0],data[1],data[2]); }
		EAVL_HOSTDEVICE float		  getSurfaceArea() 	const  { return 4.f*data[3]*data[3]*3.1415926535f;}
		EAVL_HOSTDEVICE float 		  operator[](int i) const 
		{
			return data[i];
		}

};


#endif