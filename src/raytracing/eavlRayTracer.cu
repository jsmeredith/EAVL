#include <eavlRayTracer.h>


eavlRayTracer::eavlRayTracer()
{
	//create default color map
	float *defaultColorMap = new float[8];
	for (int i = 0; i < 8; ++i)
	{
		defaultColorMap[i] = 1.f;
	}
	colorMap = new eavlTextureObject<float>(8, defaultColorMap, true);
}

eavlRayTracer::~eavlRayTracer()
{
	delete colorMap;
}


void eavlRayTracer::setColorMap3f(float* cmap, const int &nColors)
{
	if(nColors < 1)
	{
		THROW(eavlException, "Cannot set color map size of less than 1");
	}
    delete colorMap;
    colorMap = new eavlTextureObject<float>(nColors, cmap, false);
}
