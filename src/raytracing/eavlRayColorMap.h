#ifndef EAVL_RAY_COLOR_MAP_H
#define EAVL_RAY_COLOR_MAP_H
#include <vector>

struct 

class eavlRayColorMap
{
	protected:
		int  colorMapSize;
		int  numPegPoints;
		bool hasAlpha;
		float *rgba;
	public:
		
		eavlRayColorMap()
		{
			colorMapSize = 1024;
			numPegPoints = 0;
			hasAlpha = false;
		}
		eavlRayColorMap(const int _colorMapSize, _hasAlpha)
		{
			if(_colorMapSize < 1) THROW(eavlException, "Color map size must be at least 1");
			colorMapSize = _colorMapSize;
			hasAlpha = _hasAlpha;
			numPegPoints = 0;
		}

		void addRGBPegPoint(float r, float g, float b, float value)
		{

		}

		void addAlphaPegPoint(float alpha, float value)
		{
			if(alpha < 0 || alpha > 1) THROW(eavlException, "Alpha value must be normailized.");
		}


};
#endif