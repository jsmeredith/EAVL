
#include "eavlRayCamera.h"
#include "eavlCountingIterator.h"
#include "eavlRadixSortOp.h"
#include "eavlMapOp.h"
#include "eavlSampler.h"

EAVL_HOSTONLY eavlRayCamera::eavlRayCamera()
{
	height = 100;
	width = 100;
	fovx = 30.f;
	fovy = 30.f;
	zoom = 1.0f;

	isViewDirty = true;
	isResDirty = true;

	look.x = 0.f;
	look.y = 0.f;
	look.z = 1.f;

	up.x = 0.f;
	up.y = 1.f;
	up.z = 0.f;
  lookAtSet = false;
	size = height * width;
	mortonSort = true;

	pixelIndexes = new eavlIntArray("",1,size);
};

EAVL_HOSTONLY eavlRayCamera::~eavlRayCamera()
{
	delete pixelIndexes;
};


struct MortonRayFunctor
{
	int height;
	int width;
	float fwidth;
	float fheight;
	MortonRayFunctor(int _height, int _width)
	{
		height = _height;
		width = _width;
		fheight = (float) height;
		fwidth = (float) width;
	}
	EAVL_FUNCTOR tuple<int> operator()(tuple<int>  idx)
	{												
		
		float w = (float)(idx%width)/fwidth;
        float h = (float)(idx/width)/fheight;

		unsigned int code = morton2D(w,h); 
        return tuple<int>(code);
	} 
};

struct PerspectiveRayGenFunctor
{
    int w;
    int h; 
    eavlVector3 nlook;// normalized look
    eavlVector3 delta_x;
    eavlVector3 delta_y;

    PerspectiveRayGenFunctor(int width,
    				 		 int height, 
    				 		 float half_fovX, 
    				 		 float half_fovY, 
    				 		 eavlVector3 look, 
    				 		 eavlVector3 up, 
    				 		 float _zoom)
        : w(width), h(height)
    {
        float thx = tan(half_fovX*PI/180);
        float thy = tan(half_fovY*PI/180);


        eavlVector3 ru = up%look;
        ru.normalize();

        eavlVector3 rv = ru%look;
        rv.normalize();

        delta_x = ru*(2*thx/(float)w);
        delta_y = rv*(2*thy/(float)h);
        
        if(_zoom > 0)
        {
            delta_x /= _zoom;
            delta_y /= _zoom;    
        }
        

        nlook.x = look.x;
        nlook.y = look.y;
        nlook.z = look.z;
        nlook.normalize();

    }

    EAVL_FUNCTOR tuple<float,float, float> operator()(int idx){
        int i=idx%w;
        int j=idx/w;

        eavlVector3 ray_dir=nlook+delta_x*((2*i-w)/2.0f)+delta_y*((2*j-h)/2.0f);
        ray_dir.normalize();

        return tuple<float,float,float>(ray_dir.x,ray_dir.y,ray_dir.z);

    }

};

struct PerspectiveSubsetRayGenFunctor
{
    int w;
    int h; 
    int subsetHeight;
    int subsetWidth;
    int xmin;
    int ymin;
    eavlVector3 nlook;// normalized look
    eavlVector3 delta_x;
    eavlVector3 delta_y;

    PerspectiveSubsetRayGenFunctor(int width,
    				 		 int height, 
    				 		 float half_fovX, 
    				 		 float half_fovY, 
    				 		 eavlVector3 look, 
    				 		 eavlVector3 up, 
    				 		 float _zoom,
    				 		 int _subsetWidth,
    				 		 int _subsetHeight,
    				 		 int _xmin,
    				 		 int _ymin
    				 		 )
        : w(width), h(height)
    {
        float thx = tan(half_fovX*PI/180);
        float thy = tan(half_fovY*PI/180);
        subsetHeight = _subsetHeight;
        subsetWidth = _subsetWidth;
        xmin = _xmin;
        ymin = _ymin;
        eavlVector3 ru = up%look;
        ru.normalize();

        eavlVector3 rv = ru%look;
        rv.normalize();

        delta_x = ru*(2*thx/(float)w);
        delta_y = rv*(2*thy/(float)h);
        
        if(_zoom > 0)
        {
            delta_x /= _zoom;
            delta_y /= _zoom;    
        }
        

        nlook.x = look.x;
        nlook.y = look.y;
        nlook.z = look.z;
        nlook.normalize();

    }

    EAVL_FUNCTOR tuple<float,float, float> operator()(int idx){
        int i=idx%subsetWidth;
        int j=idx/subsetWidth;
        //add the pixel offset
        i += xmin;
        j += ymin;
        eavlVector3 ray_dir=nlook+delta_x*((2*i-w)/2.0f)+delta_y*((2*j-h)/2.0f);
        ray_dir.normalize();

        return tuple<float,float,float>(ray_dir.x,ray_dir.y,ray_dir.z);

    }

};

struct PerspectiveJitterRayGenFunctor
{
    float w;
    float h; 
    eavlVector3 nlook;// normalized look
    eavlVector3 delta_x;
    eavlVector3 delta_y;
    int sampleNum;

    PerspectiveJitterRayGenFunctor(int width,
                 int height, 
                 float half_fovX, 
                 float half_fovY, 
                 eavlVector3 look, 
                 eavlVector3 up, 
                 float _zoom,
                 int _sampleNum)
        : w(width), h(height)
    {
        float thx = tan(half_fovX*PI/180);
        float thy = tan(half_fovY*PI/180);


        eavlVector3 ru = up%look;
        ru.normalize();

        eavlVector3 rv = ru%look;
        rv.normalize();

        delta_x = ru*(2*thx/(float)w);
        delta_y = rv*(2*thy/(float)h);
        if(_zoom > 0)
        {
            delta_x /= _zoom;
            delta_y /= _zoom;    
        }
        

        nlook.x = look.x;
        nlook.y = look.y;
        nlook.z = look.z;
        nlook.normalize();
        sampleNum = _sampleNum;
    }

    EAVL_FUNCTOR tuple<float,float, float> operator()(tuple<int,int>input){
        int idx = get<0>(input);
        int seed = get<1>(input);
        float i=idx%int(w);
        float j=idx/int(w);
        float xy[2];
        eavlSampler::halton2D<3>(sampleNum+seed, xy);
        xy[0]-=.5f;
        xy[1]-=.5f;
        i+=xy[0];
        j+=xy[1];  
        eavlVector3 ray_dir=nlook+delta_x* ((2.0f*i-w)/2.0f)+delta_y*((2.0f*j-h)/2.0f);
        ray_dir.normalize();

        return tuple<float,float,float>(ray_dir.x,ray_dir.y,ray_dir.z);

    }

};

struct PerspectiveDOFRayGenFunctor
{
    float w;
    float h; 
    eavlVector3 nlook;// normalized look
    eavlVector3 delta_x;
    eavlVector3 delta_y;
    eavlVector3 lookat;
    eavlVector3 apertureXRadius;
    eavlVector3 apertureYRadius;
    int sampleNum;
    float focalDistance;
    eavlVector3 position;
    PerspectiveDOFRayGenFunctor(int width,
                 int height, 
                 float half_fovX, 
                 float half_fovY, 
                 eavlVector3 look, 
                 eavlVector3 up, 
                 float _zoom,
                 int _sampleNum,
                 eavlVector3 _lookAt,
                 eavlVector3 _position,
                 float apertureRadius)
        : w(width), h(height)
    {
        lookat = _lookAt;
        position = _position;
        eavlVector3 dist = _lookAt - position;
        focalDistance = sqrt(dist * dist);
        float thx = tan(half_fovX*PI/180) * focalDistance;
        float thy = tan(half_fovY*PI/180) * focalDistance;
        
        eavlVector3 ru = up%look;
        ru.normalize();
        eavlVector3 rv = ru%look;
        rv.normalize();

        apertureXRadius = ru *apertureRadius;
        apertureYRadius = rv *apertureRadius;

        delta_x = ru*(2*thx/(float)w);
        delta_y = rv*(2*thy/(float)h);
        
        if(_zoom > 0)
        {
            delta_x /= _zoom;
            delta_y /= _zoom;    
        }
        

        nlook.x = look.x;
        nlook.y = look.y;
        nlook.z = look.z;
        nlook.normalize();
        sampleNum = _sampleNum;
    }

    EAVL_FUNCTOR tuple<float,float, float> operator()(tuple<int,int>input){
        int idx = get<0>(input);
        int seed = get<1>(input);
        float i=idx%int(w);
        float j=idx/int(w);
        float xy[2];
        eavlSampler::halton2D<3>(sampleNum+seed, xy);
      
        xy[0]= xy[0]*2.f - 1.f;
        xy[1]= xy[1]*2.f - 1.f;

        eavlVector3 randomLensPoint = position + apertureXRadius * xy[0] + apertureYRadius * xy[1];

        eavlVector3 imagePoint = lookat + delta_x* ((2.0f*i-w)/2.0f)+delta_y*((2.0f*j-h)/2.0f);

        eavlVector3 ray_dir =  imagePoint - randomLensPoint;

        ray_dir.normalize();

        return tuple<float,float,float>(ray_dir.x,ray_dir.y,ray_dir.z);

    }

};

EAVL_HOSTONLY void eavlRayCamera::generatePixelIndexes()
{
	eavlCountingIterator::generateIterator(pixelIndexes);
	if(mortonSort)
	{
		eavlIntArray* mortonCodes = new eavlIntArray("",1, size);

		eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(pixelIndexes), //dummy arg
                                                 eavlOpArgs(mortonCodes),
                                                 MortonRayFunctor(height, width)),
                                                 "init");
    eavlExecutor::Go();

	  eavlExecutor::AddOperation(new_eavlRadixSortOp(eavlOpArgs(mortonCodes),
                                                   eavlOpArgs(pixelIndexes), false),
                                                   "");
    eavlExecutor::Go();
    delete mortonCodes;
	}

}

EAVL_HOSTONLY void eavlRayCamera::createRaysSubset(eavlRay* rays, int &xmin, int &ymin, int &dx, int &dy)
{
  //this is a subset of the current size.
  int subsetSize = dx * dy;
 if( !isResDirty && !isViewDirty && rays->numRays == subsetSize) 
    { 
      cerr<<"No rays to create\n"; 
      eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(rays->hitIdx), //dummy arg
                                              eavlOpArgs(rays->hitIdx),
                                              IntMemsetFunctor(-1)),
                                              "init");
      eavlExecutor::Go();
      return;
    }

    if(isResDirty || rays->numRays != subsetSize)
    {
    	if(isResDirty)
    	{
    		delete pixelIndexes;
    		pixelIndexes =  new eavlIntArray("",1,subsetSize);
    	}

    	if(rays->numRays != subsetSize) rays->resize(subsetSize);
    	generatePixelIndexes();
    }
    isResDirty = false;
    eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(pixelIndexes), //dummy arg
                                             eavlOpArgs(rays->rayOriginX,rays->rayOriginY,rays->rayOriginZ),
                                             FloatMemsetFunctor3to3(position.x,position.y,position.z)),
                                             "init");
    eavlExecutor::Go();
    if(lookAtSet)
    {
      look = lookat - position;
      look.normalize();  
    }
    
    eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(pixelIndexes),
                                             eavlOpArgs(rays->rayDirX ,rays->rayDirY, rays->rayDirZ),
                                             PerspectiveSubsetRayGenFunctor(width, 
                                                                            height, 
                                                                            fovx, 
                                                                            fovy, 
                                                                            look, 
                                                                            up, 
                                                                            zoom,
                                                                            dx,
                                                                            dy,
                                                                            xmin,
                                                                            ymin)),
                                             "ray gen");
    eavlExecutor::Go();

    eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(rays->hitIdx), //dummy arg
                                             eavlOpArgs(rays->hitIdx),
                                             IntMemsetFunctor(0)),
                                             "init");
    eavlExecutor::Go();
}

EAVL_HOSTONLY void eavlRayCamera::createRays(eavlRay* rays)
{
    if( !isResDirty && !isViewDirty && rays->numRays == size) 
    { 
      cerr<<"No rays to create\n"; 
      eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(rays->hitIdx), //dummy arg
                                              eavlOpArgs(rays->hitIdx),
                                              IntMemsetFunctor(-1)),
                                              "init");
      eavlExecutor::Go();
      return;
    }

    if(isResDirty || rays->numRays != size)
    {
    	if(isResDirty)
    	{
    		delete pixelIndexes;
    		pixelIndexes =  new eavlIntArray("",1,size);
    	}

    	if(rays->numRays != size) rays->resize(size);
    	generatePixelIndexes();
    }
    isResDirty = false;

    eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(pixelIndexes), //dummy arg
                                             eavlOpArgs(rays->rayOriginX,rays->rayOriginY,rays->rayOriginZ),
                                             FloatMemsetFunctor3to3(position.x,position.y,position.z)),
                                             "init");
    eavlExecutor::Go();
    if(lookAtSet)
    {
      look = lookat - position;
      look.normalize();  
    }
    
    eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(pixelIndexes),
                                             eavlOpArgs(rays->rayDirX ,rays->rayDirY, rays->rayDirZ),
                                             PerspectiveRayGenFunctor(width, height, fovx, fovy, look, up, zoom)),
                                             "ray gen");
    eavlExecutor::Go();

    eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(rays->hitIdx), //dummy arg
                                             eavlOpArgs(rays->hitIdx),
                                             IntMemsetFunctor(0)),
                                             "init");
    eavlExecutor::Go();
}  


EAVL_HOSTONLY void eavlRayCamera::createJitterRays(eavlRay* rays, eavlIntArray * seeds, int sampleNum)
{
    if( !isResDirty && !isViewDirty && rays->numRays == size) 
    { 
      cerr<<"No rays to create\n"; 
      eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(rays->hitIdx), //dummy arg
                                              eavlOpArgs(rays->hitIdx),
                                              IntMemsetFunctor(-1)),
                                              "init");
      eavlExecutor::Go();
      return;
    }

    if(isResDirty || rays->numRays != size)
    {
      if(isResDirty)
      {
        delete pixelIndexes;
        pixelIndexes =  new eavlIntArray("",1,size);
      }

      if(rays->numRays != size) rays->resize(size);
      generatePixelIndexes();
    }
    isResDirty = false;

    eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(pixelIndexes), //dummy arg
                                             eavlOpArgs(rays->rayOriginX,rays->rayOriginY,rays->rayOriginZ),
                                             FloatMemsetFunctor3to3(position.x,position.y,position.z)),
                                             "init");
    eavlExecutor::Go();
    if(lookAtSet)
    {
      look = lookat - position;
      look.normalize();  
    }
    
    eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(pixelIndexes,seeds),
                                             eavlOpArgs(rays->rayDirX ,rays->rayDirY, rays->rayDirZ),
                                             PerspectiveJitterRayGenFunctor(width, height, fovx, fovy, look, up, zoom, sampleNum)),
                                             "ray gen");
    eavlExecutor::Go();

    eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(rays->hitIdx), //dummy arg
                                             eavlOpArgs(rays->hitIdx),
                                             IntMemsetFunctor(0)),
                                             "init");
    eavlExecutor::Go();
}  


EAVL_HOSTONLY void eavlRayCamera::createDOFRays(eavlRay* rays, eavlIntArray * seeds, int sampleNum, float apertureSize)
{
    if( !isResDirty && !isViewDirty && rays->numRays == size) 
    { 
      cerr<<"No rays to create\n"; 
      eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(rays->hitIdx), //dummy arg
                                              eavlOpArgs(rays->hitIdx),
                                              IntMemsetFunctor(-1)),
                                              "init");
      eavlExecutor::Go();
      return;
    }

    if(isResDirty || rays->numRays != size)
    {
      if(isResDirty)
      {
        delete pixelIndexes;
        pixelIndexes =  new eavlIntArray("",1,size);
      }

      if(rays->numRays != size) rays->resize(size);
      generatePixelIndexes();
    }
    isResDirty = false;

    eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(pixelIndexes), //dummy arg
                                             eavlOpArgs(rays->rayOriginX,rays->rayOriginY,rays->rayOriginZ),
                                             FloatMemsetFunctor3to3(position.x,position.y,position.z)),
                                             "init");
    eavlExecutor::Go();
    if(lookAtSet)
    {
      look = lookat - position;
      look.normalize();  
    }
    
    eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(pixelIndexes,seeds),
                                             eavlOpArgs(rays->rayDirX ,rays->rayDirY, rays->rayDirZ),
                                             PerspectiveDOFRayGenFunctor(width, height, fovx, fovy, look, up, zoom, sampleNum, lookat, position, apertureSize)),
                                             "ray gen");
    eavlExecutor::Go();

    eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(rays->hitIdx), //dummy arg
                                             eavlOpArgs(rays->hitIdx),
                                             IntMemsetFunctor(0)),
                                             "init");
    eavlExecutor::Go();
}  

EAVL_HOSTONLY bool  eavlRayCamera::getIsViewDirty()
{
	return isViewDirty;
}

EAVL_HOSTONLY void eavlRayCamera::setWidth(const int &_width)
{
	if( _width > 0)
	{
		if(width != _width) isResDirty = true;
		width = _width;
		size = height * width;
	}
	else printf("Error: Cannot set width to less than 1\n");
};

EAVL_HOSTONLY void eavlRayCamera::setHeight(const int &_height)
{
	if( _height > 0)
	{
		if(height != _height) isResDirty = true;
		height = _height;
		size = height * width;
	}
	else printf("Error: Cannot set height to less than 1\n");
};

EAVL_HOSTONLY int eavlRayCamera::getWidth()
{
	return width;
};

EAVL_HOSTONLY int eavlRayCamera::getHeight()
{
	return height;
};

EAVL_HOSTONLY void eavlRayCamera::setFOVX(const float &_fovx)
{
	if( _fovx > 0.f)
	{
		if(_fovx != fovx) isViewDirty = true;
		fovx = _fovx;
	}
	else printf("Error: Cannot set fovx less than 1\n");
};

EAVL_HOSTONLY void eavlRayCamera::setFOVY(const float &_fovy)
{
	if( _fovy > 0.f)
	{
		if(_fovy != fovy) isViewDirty = true;
		fovy = _fovy;
	}
	else printf("Error: Cannot set fovy less than 1\n");
};

EAVL_HOSTONLY float eavlRayCamera::getFOVX()
{
	return fovx;
};

EAVL_HOSTONLY float eavlRayCamera::getFOVY()
{
	return fovy;
};

EAVL_HOSTONLY void eavlRayCamera::setCameraPosition(const float &_x, const float &_y, const float &_z)
{
  if( position.x != _x) isViewDirty=true;
  position.x = _x;
  if( position.y != _y) isViewDirty=true;
  position.y = _y;
  if( position.z != _z) isViewDirty=true;
  position.z = _z;
}

EAVL_HOSTONLY void eavlRayCamera::setMortonSorting(bool on)
{
	mortonSort = on;
}

EAVL_HOSTONLY bool eavlRayCamera::getMortonSorting()
{
  return mortonSort;
}

EAVL_HOSTONLY void eavlRayCamera::setCameraPositionX(const float &_x)
{
  if( position.x != _x) isViewDirty=true;
  position.x = _x;
}

EAVL_HOSTONLY void eavlRayCamera::setCameraPositionY(const float &_y)
{
  if( position.y != _y) isViewDirty=true;
  position.x = _y;
}

EAVL_HOSTONLY void eavlRayCamera::setCameraPositionZ(const float &_z)
{
  if( position.z != _z) isViewDirty=true;
  position.z = _z;
}

EAVL_HOSTONLY float eavlRayCamera::getCameraPositionX()
{
  return position.x;
}

EAVL_HOSTONLY float eavlRayCamera::getCameraPositionY()
{
  return position.y;
}

EAVL_HOSTONLY float eavlRayCamera::getCameraPositionZ()
{
  return position.z;
}



EAVL_HOSTONLY void eavlRayCamera::setCameraUp(const float &_x, const float &_y, const float &_z)
{
  if( up.x != _x) isViewDirty=true;
  up.x = _x;
  if( up.y != _y) isViewDirty=true;
  up.y = _y;
  if( up.z != _z) isViewDirty=true;
  up.z = _z;
}

EAVL_HOSTONLY void eavlRayCamera::setCameraUpX(const float &_x)
{
  if( up.x != _x) isViewDirty=true;
  up.x = _x;
}

EAVL_HOSTONLY void eavlRayCamera::setCameraUpY(const float &_y)
{
  if( up.y != _y) isViewDirty=true;
  up.y = _y;
}

EAVL_HOSTONLY void eavlRayCamera::setCameraUpZ(const float &_z)
{
  if( up.z != _z) isViewDirty=true;
  up.z = _z;
}

EAVL_HOSTONLY float eavlRayCamera::getCameraUpX()
{
  return up.x;
}

EAVL_HOSTONLY float eavlRayCamera::getCameraUpY()
{
  return up.y;
}

EAVL_HOSTONLY float eavlRayCamera::getCameraUpZ()
{
  return up.z;
}

EAVL_HOSTONLY void eavlRayCamera::setCameraLook(const float &_x, const float &_y, const float &_z)
{
  lookAtSet = false;
  if( look.x != _x) isViewDirty=true;
  look.x = _x;
  if( look.y != _y) isViewDirty=true;
  look.y = _y;
  if( look.z != _z) isViewDirty=true;
  look.z = _z;
}

EAVL_HOSTONLY void eavlRayCamera::setCameraLookX(const float &_x)
{
  if( look.x != _x) isViewDirty=true;
  look.x = _x;
}

EAVL_HOSTONLY void eavlRayCamera::setCameraLookY(const float &_y)
{
  if( look.y != _y) isViewDirty=true;
  look.y=_y;
}

EAVL_HOSTONLY void eavlRayCamera::setCameraLookZ(const float &_z)
{
  if( look.z != _z) isViewDirty=true;
  look.z=_z;
}

EAVL_HOSTONLY float eavlRayCamera::getCameraLookX()
{
  return look.x;
}

EAVL_HOSTONLY float eavlRayCamera::getCameraLookY()
{
  return look.y;
}

EAVL_HOSTONLY float eavlRayCamera::getCameraLookZ()
{
  return look.z;
}

EAVL_HOSTONLY eavlIntArray* eavlRayCamera::getPixelIndexes()
{
  return pixelIndexes;
} 

EAVL_HOSTONLY void eavlRayCamera::setCameraZoom(const float &_zoom)
{
  if(zoom != _zoom) isViewDirty = true;
  zoom = _zoom;
} 

EAVL_HOSTONLY void eavlRayCamera::lookAtPosition(const float &_x, const float &_y,const float &_z)
{
  lookat.x = _x;
  lookat.y = _y;
  lookat.z = _z;
  lookAtSet = true;
}

EAVL_HOSTONLY void eavlRayCamera::printSummary()
{
	cerr<<"----------------Camera Parameters--------------"<<endl;
	cerr<<"Height      : "<<height<<endl;
	cerr<<"Width       : "<<width<<endl;
	cerr<<"FovX        : "<<fovx<<endl;
	cerr<<"FovY        : "<<fovy<<endl;
	cerr<<"Zoom        : "<<zoom<<endl;
	cerr<<"Morton sort : "<<mortonSort<<endl;
	cerr<<"Look        : "<<look<<endl;
	cerr<<"Look at     : "<<lookat<<endl;
	cerr<<"Up          : "<<up<<endl;
	cerr<<"Position    : "<<position<<endl;
	cerr<<"--------------------------------------------------"<<endl;
	
}
