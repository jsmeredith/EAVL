#ifndef EAVL_VOLUME_RENDERER_MUTATOR_H
#define EAVL_VOLUME_RENDERER_MUTATOR_H
#include "eavlFilter.h"

struct Camera
{
	eavlVector3 	look;
    eavlVector3 	lookat;
    eavlVector3		position;
    eavlVector3		up;
    float 			fovx;
    float 			fovy;
    float			zoom;
};



class eavlVolumeRendererMutator : public eavlMutator
{
  public:
    eavlVolumeRendererMutator();
    void SetField(const string &name)
    {
        fieldname = name;
    }
    virtual void Execute();
  protected:
    string fieldname;
    
    int 	height;
    int 	width;


    Camera  camera



};
#endif