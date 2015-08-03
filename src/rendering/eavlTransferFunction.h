#ifndef EAVL_TRANSFER_FUNCTION
#define EAVL_TRANSFER_FUNCTION
#include "eavlColorTable.h"

class eavlTransferFunction
{
protected:
	eavlColorTable *rgb;
	eavlColorTable *alpha;
public:
	eavlTransferFunction()
	{
		rgb = new eavlColorTable(true);
		alpha = new eavlColorTable(true);
	}
	
	eavlTransferFunction(string rgbName)
	{
		rgb = new eavlColorTable(rgbName);
		alpha = new eavlColorTable();
	}
	
	~eavlTransferFunction()
	{
		delete rgb;
		delete alpha;
	}
	
	void AddRGBControlPoint(double position, eavlColor color)
	{
		rgb->AddControlPoint(position, color);
	}
	
	void Clear()
	{
		rgb->Clear();
		alpha->Clear();
	}
	
	void operator=(const eavlTransferFunction &tf)
    {
        rgb = tf.rgb;
        alpha = tf.alpha;
    }
    
    void SetByColorTableName(string ctname)
    {
    	delete rgb;
    	rgb = new eavlColorTable( ctname);
    }
	
	void AddAlphaControlPoint(double position, double alphaVal)
	{
		alpha->AddControlPoint(position, eavlColor(alphaVal, alphaVal, alphaVal));
	}
	void CreateDefaultAlpha()
	{
		alpha->AddControlPoint(0.0f, eavlColor(0.0f, 0.0f, 0.0f));
		alpha->AddControlPoint(0.2f, eavlColor(0.2f, 0.2f, 0.2f));
		alpha->AddControlPoint(0.4f, eavlColor(0.4f, 0.4f, 0.4f));
		alpha->AddControlPoint(0.6f, eavlColor(0.6f, 0.6f, 0.6f));
		alpha->AddControlPoint(0.8f, eavlColor(0.8f, 0.8f, 0.8f));
		alpha->AddControlPoint(1.0f, eavlColor(1.0f, 1.f, 1.f));
	}
	void GetTransferFunction(int n, float *transferFunction)
	{	
		float *rgbValues = new float[n*3];
		float *alphaValues = new float[n*3];
		rgb->Sample(n, rgbValues);
		alpha->Sample(n, alphaValues);
		cout<<"Alphas : "<<endl;
		for(int i = 0; i < n; i++)
		{
			cout<<rgbValues[i*3+0]<< " "<<rgbValues[i*3+1]<<" "<<rgbValues[i*3+2]<<endl;
		}
		cout<<endl;
		for(int i = 0; i < n; i++)
		{
			transferFunction[i*4+0] = rgbValues[i*3+0];
			transferFunction[i*4+1] = rgbValues[i*3+1];
			transferFunction[i*4+2] = rgbValues[i*3+2];
			transferFunction[i*4+3] = alphaValues[i*3+0];
		}
		delete[] rgbValues;
		delete[] alphaValues;
	}
	 //
	 //  When using a custom transfer function.
	 //  the color bar needs to render correctly
	 //
	eavlColorTable * GetRGBColorTable()
	{
	  return rgb;
	}

};
#endif
