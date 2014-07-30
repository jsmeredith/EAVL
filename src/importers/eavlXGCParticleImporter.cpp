// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.

#include "eavlXGCParticleImporter.h"
#include "eavlCellSetAllPoints.h" 

eavlXGCParticleImporter::eavlXGCParticleImporter(const string &filename)
{
    timestep = 0;
    maxnum = 0;
    enphase = 0;
    inphase = 0;
    emaxgid = 0;
    imaxgid = 0;
    nvars = 0;
    time = 0;
    fp = NULL;
    
    string::size_type i0 = filename.rfind("xgc.");
    string::size_type i1 = filename.rfind(".bp");
    
#ifdef PARALLEL
    fp = adios_read_open_file(filename.c_str(), ADIOS_READ_METHOD_BP, (MPI_Comm)VISIT_MPI_COMM);
#else
    MPI_Comm comm_dummy = 0;
    fp = adios_read_open_file(filename.c_str(), ADIOS_READ_METHOD_BP, comm_dummy);
#endif
    
    if(fp == NULL)
		THROW(eavlException, "XGC variable file not found.");

    Initialize();
}

//Reads a staged adios file
eavlXGCParticleImporter::eavlXGCParticleImporter(	const string &filename, 
													ADIOS_READ_METHOD method, 
													MPI_Comm comm, 
													ADIOS_LOCKMODE mode, 
													int timeout_sec
												)
{
    timestep = 0;
    maxnum = 0;
    enphase = 0;
    inphase = 0;
    emaxgid = 0;
    imaxgid = 0;
    nvars = 0;
    time = 0;
    fp = NULL;
    
    string::size_type i0 = filename.rfind("xgc.");
    string::size_type i1 = filename.rfind(".bp");
    
    fp = adios_read_open(filename.c_str(), method, comm, mode, timeout_sec);
    
    if(fp == NULL)
		THROW(eavlException, "XGC variable file not found.");

    Initialize();
}


eavlXGCParticleImporter::~eavlXGCParticleImporter()
{
    if(fp)
        adios_read_close(fp);
    fp = NULL;

    map<string, ADIOS_VARINFO*>::const_iterator it;
    for(it = ephase.begin(); it != ephase.end(); it++)
		adios_free_varinfo(it->second);
    ephase.clear();
    for(it = iphase.begin(); it != iphase.end(); it++)
		adios_free_varinfo(it->second);
    iphase.clear();
    for(it = egid.begin(); it != egid.end(); it++)
		adios_free_varinfo(it->second);
    egid.clear();
    for(it = igid.begin(); it != igid.end(); it++)
		adios_free_varinfo(it->second);
    igid.clear();
}

void
eavlXGCParticleImporter::Initialize()
{
    ephase.clear();
    iphase.clear();
    egid.clear();
    igid.clear();
    
    nvars = fp->nvars;
    
    for(int i = 0; i < fp->nvars; i++)
    {
    	ADIOS_VARINFO *avi = adios_inq_var_byid(fp, i);
		string longvarNm(&fp->var_namelist[i][1]);
		string varNm = longvarNm.substr(longvarNm.find("/",1,1)+1,longvarNm.length());
	    string nodeNum = longvarNm.substr(longvarNm.find("_",1,1)+1, 5);	
	
		if(varNm == "ephase") 
		{
			ephase[longvarNm] = avi;
		} 
		else if(varNm == "egid") 
		{
			egid[longvarNm] = avi;
		} 
		else if(varNm == "iphase") 
		{
			iphase[longvarNm] = avi;
		}
		else if(varNm == "igid") 
		{
			igid[longvarNm] = avi;
		} 
		else if(i < 13) 
		{
			if (varNm == "timestep") 
			{
				timestep = (int)(*(int *)avi->value);
			} 
			else if(varNm == "time") 
			{
				time = (double)(*(double *)avi->value);
			} 
			else if(varNm == "maxnum") 
			{
				maxnum = (int)(*(int *)avi->value);
			} 
			else if (varNm == "inphase") 
			{
				inphase = (int)(*(int *)avi->value);
			} 
			else if(varNm == "enphase") 
			{
				enphase = (int)(*(int *)avi->value);
			} 
			else if(varNm == "emaxgid") 
			{
				emaxgid = (long long)(*(long long *)avi->value);
			} 
			else if(varNm == "imaxgid") 
			{
				imaxgid = (long long)(*(long long *)avi->value);
			}
			adios_free_varinfo(avi);
		} 
		else 
		{
			adios_free_varinfo(avi);
		}		
	}
}

vector<string>
eavlXGCParticleImporter::GetMeshList()
{
    vector<string> m;
    m.push_back("iMesh");
    m.push_back("eMesh");
    return m;
}

vector<string>
eavlXGCParticleImporter::GetCellSetList(const std::string &mesh)
{
    vector<string> m;
    if(mesh == "iMesh")
		m.push_back("iMesh_cells");
    else if(mesh == "eMesh")
		m.push_back("eMesh_cells");
    
    return m;
}

vector<string>
eavlXGCParticleImporter::GetFieldList(const std::string &mesh)
{
    vector<string> fields;
    if(mesh == "iMesh") 
    {
    	fields.push_back("iphase");
    	fields.push_back("igid");
    }
    else if(mesh == "eMesh") 
    {
    	fields.push_back("ephase");
    	fields.push_back("egid");
    }

    return fields;
}

eavlDataSet *
eavlXGCParticleImporter::GetMesh(const string &name, int chunk)
{
    eavlDataSet *ds = new eavlDataSet;
    if(name == "iMesh") 
    {
    	ds->SetNumPoints(imaxgid);
		eavlCoordinatesCartesian *coords = new eavlCoordinatesCartesian(NULL,
										eavlCoordinatesCartesian::X,
										eavlCoordinatesCartesian::Y,
										eavlCoordinatesCartesian::Z);
		ds->AddCoordinateSystem(coords);
		coords->SetAxis(0, new eavlCoordinateAxisField("xcoords", 0));
		coords->SetAxis(1, new eavlCoordinateAxisField("ycoords", 0));
		coords->SetAxis(2, new eavlCoordinateAxisField("zcoords", 0));
		
		eavlArray *axisValues[3] = {
									new eavlFloatArray("xcoords", 1),
									new eavlFloatArray("ycoords", 1),
									new eavlFloatArray("zcoords", 1)
									};
		axisValues[0]->SetNumberOfTuples(imaxgid);
		axisValues[1]->SetNumberOfTuples(imaxgid);
		axisValues[2]->SetNumberOfTuples(imaxgid);

	
		//Set all of the axis values to the x, y, z coordinates of the 
		//iphase particles; set computational node origin
		eavlIntArray *originNode = new eavlIntArray("originNode", 1, imaxgid);
		
		uint64_t s[3], c[3];
		double *buff, R, Z, phi;
		int nt = 1, idx = 0, retval;
		map<string, ADIOS_VARINFO*>::const_iterator it;
		for(it = iphase.begin(); it != iphase.end(); it++) 
		{
			ADIOS_SELECTION *sel = MakeSelection(it->second, s, c);
			nt = 1;
			for (int i = 0; i < it->second->ndim; i++)
				nt *= c[i];
			
			buff = new double[nt];
			adios_schedule_read_byid(fp, sel, it->second->varid, 0, 1, buff);
			retval = adios_perform_reads(fp, 1);
			adios_selection_delete(sel);
			
			string nodeNum = it->first.substr(it->first.find("_",1,1)+1, 5);
			
			for(int i = 0; i < nt; i+=9) 
			{
				R = buff[i];
				Z = buff[i+1];
				phi = buff[i+2];
				axisValues[0]->SetComponentFromDouble(idx, 0, R*cos(phi));
				axisValues[1]->SetComponentFromDouble(idx, 0, R*sin(phi));
				axisValues[2]->SetComponentFromDouble(idx, 0, Z);
				originNode->SetValue(idx, atoi(nodeNum.c_str()));
				idx++;
			}
			delete [] buff;
		}
		
		ds->AddField(new eavlField(1, axisValues[0], eavlField::ASSOC_POINTS));
		ds->AddField(new eavlField(1, axisValues[1], eavlField::ASSOC_POINTS));
		ds->AddField(new eavlField(1, axisValues[2], eavlField::ASSOC_POINTS));
		ds->AddField(new eavlField(1, originNode, eavlField::ASSOC_POINTS));
		
		eavlCellSet *cellSet = new eavlCellSetAllPoints(name + "_I_Cells", imaxgid);
		ds->AddCellSet(cellSet);
		//-- END set axis values
	
		//----Set the ids of all axis values
		idx = 0;
		long long *idBuff;
		eavlIntArray *axisIds = new eavlIntArray("id", 1, imaxgid);
		for(it = igid.begin(); it != igid.end(); it++) 
		{
			ADIOS_SELECTION *sel = MakeSelection(it->second, s, c);
			nt = 1;
			for (int i = 0; i < it->second->ndim; i++)
				nt *= c[i];
			
			idBuff = new long long[nt];
			adios_schedule_read_byid(fp, sel, it->second->varid, 0, 1, idBuff);
			retval = adios_perform_reads(fp, 1);
			adios_selection_delete(sel);

			for(int i = 0; i < nt; i++) 
			{
				axisIds->SetValue(idx, (int)idBuff[i]);
				idx++;
			}
			delete [] idBuff;
		}
		ds->AddField(new eavlField(1, axisIds, eavlField::ASSOC_POINTS));
		//-- END set ids
		
    } 
    else if(name == "eMesh") 
    {
    	ds->SetNumPoints(emaxgid);
		eavlCoordinatesCartesian *coords = new eavlCoordinatesCartesian(NULL,
												eavlCoordinatesCartesian::X,
												eavlCoordinatesCartesian::Y,
												eavlCoordinatesCartesian::Z);
		ds->AddCoordinateSystem(coords);
		coords->SetAxis(0, new eavlCoordinateAxisField("xcoords", 0));
		coords->SetAxis(1, new eavlCoordinateAxisField("ycoords", 0));
		coords->SetAxis(2, new eavlCoordinateAxisField("zcoords", 0));
		
		eavlArray *axisValues[3] = {
									new eavlFloatArray("xcoords", 1),
									new eavlFloatArray("ycoords", 1),
									new eavlFloatArray("zcoords", 1)
									};
		axisValues[0]->SetNumberOfTuples(emaxgid);
		axisValues[1]->SetNumberOfTuples(emaxgid);
		axisValues[2]->SetNumberOfTuples(emaxgid);

		//Set all of the axis values to the x, y, z coordinates of the 
		//iphase particles; set computational node origin
		eavlIntArray *originNode = new eavlIntArray("originNode", 1, emaxgid);
		
		uint64_t s[3], c[3];
		int nt = 1, idx = 0, retval;
		double *buff, R, Z, phi;
		map<string, ADIOS_VARINFO*>::const_iterator it;
		for(it = ephase.begin(); it != ephase.end(); it++) 
		{
			ADIOS_SELECTION *sel = MakeSelection(it->second, s, c);
			nt = 1;
			for(int i = 0; i < it->second->ndim; i++)
				nt *= c[i];
			
			buff = new double[nt];
			adios_schedule_read_byid(fp, sel, it->second->varid, 0, 1, buff);
			retval = adios_perform_reads(fp, 1);
			adios_selection_delete(sel);
			
			string nodeNum = it->first.substr(it->first.find("_",1,1)+1, 5);
			
			for(int i = 0; i < nt; i+=9) 
			{
				R = buff[i];
				Z = buff[i+1];
				phi = buff[i+2];
				axisValues[0]->SetComponentFromDouble(idx, 0, R*cos(phi));
				axisValues[1]->SetComponentFromDouble(idx, 0, R*sin(phi));
				axisValues[2]->SetComponentFromDouble(idx, 0, Z);
				originNode->SetValue(idx, atoi(nodeNum.c_str()));
				idx++;
			}
			delete [] buff;
		}
		
		ds->AddField(new eavlField(1, axisValues[0], eavlField::ASSOC_POINTS));
		ds->AddField(new eavlField(1, axisValues[1], eavlField::ASSOC_POINTS));
		ds->AddField(new eavlField(1, axisValues[2], eavlField::ASSOC_POINTS));
		ds->AddField(new eavlField(1, originNode, eavlField::ASSOC_POINTS));
		
		eavlCellSet *cellSet = new eavlCellSetAllPoints(name + "_E_Cells", emaxgid);
		ds->AddCellSet(cellSet);
		//-- END set axis values
	
	
		//----Set the ids of all axis values
		idx = 0;
		long long *idBuff;
		eavlIntArray *axisIds = new eavlIntArray("id", 1, emaxgid);
		
		for(it = egid.begin(); it != egid.end(); it++) 
		{
			ADIOS_SELECTION *sel = MakeSelection(it->second, s, c);
			nt = 1;
			for (int i = 0; i < it->second->ndim; i++)
				nt *= c[i];
			
			idBuff = new long long[nt];
			adios_schedule_read_byid(fp, sel, it->second->varid, 0, 1, idBuff);
			retval = adios_perform_reads(fp, 1);
			adios_selection_delete(sel);

			for(int i = 0; i < nt; i++) 
			{
				axisIds->SetValue(idx, (int)idBuff[i]);
				idx++;
			}
			delete [] idBuff;
		}
		ds->AddField(new eavlField(1, axisIds, eavlField::ASSOC_POINTS));
    }
    return ds;
}


eavlField *
eavlXGCParticleImporter::GetField(const string &name, const string &mesh, int chunk)
{
	int idx = 0;
	eavlField *field;
	uint64_t s[3], c[3];
	map<string, ADIOS_VARINFO*>::const_iterator it;
	if(name.compare("ephase") == 0)
	{
		eavlFloatArray *arr = new eavlFloatArray(name, 1);
		arr->SetNumberOfTuples(emaxgid*9);
		for(it = ephase.begin(); it != ephase.end(); it++)
		{	
			int nt = 1;
			ADIOS_SELECTION *sel = MakeSelection(it->second, s, c);
			for(int i = 0; i < it->second->ndim; i++)
				nt *= c[i];
			
			double *buff = new double[nt];
			adios_schedule_read_byid(fp, sel, it->second->varid, 0, 1, buff);
			int retval = adios_perform_reads(fp, 1);
			adios_selection_delete(sel);

			for(int i = 0; i < nt; i++) 
			{
				arr->SetComponentFromDouble(idx, 0, buff[i]);
				idx++;
			}
			delete [] buff;
		}
		field = new eavlField(1, arr, eavlField::ASSOC_POINTS);
	} 
	else if(name.compare("iphase") == 0)
	{
		eavlFloatArray *arr = new eavlFloatArray(name, 1);
		arr->SetNumberOfTuples(imaxgid*9);
		for(it = iphase.begin(); it != iphase.end(); it++) 
		{
			int nt = 1;
			ADIOS_SELECTION *sel = MakeSelection(it->second, s, c);
			for (int i = 0; i < it->second->ndim; i++)
				nt *= c[i];
			
			double *buff = new double[nt];
			adios_schedule_read_byid(fp, sel, it->second->varid, 0, 1, buff);
			int retval = adios_perform_reads(fp, 1);
			adios_selection_delete(sel);

			for(int i = 0; i < nt; i++) 
			{
				arr->SetComponentFromDouble(idx, 0, buff[i]);
				idx++;
			}
			delete [] buff;
		}
		field = new eavlField(1, arr, eavlField::ASSOC_POINTS);
	} 
	else if(name.compare("igid") == 0) 
	{
		eavlIntArray *arr = new eavlIntArray(name, 1);
		arr->SetNumberOfTuples(imaxgid);
		for(it = igid.begin(); it != igid.end(); it++) 
		{
			int nt = 1;
			ADIOS_SELECTION *sel = MakeSelection(it->second, s, c);
			for(int i = 0; i < it->second->ndim; i++)
				nt *= c[i];
				
			long long *buff = new long long[nt];
			adios_schedule_read_byid(fp, sel, it->second->varid, 0, 1, buff);
			int retval = adios_perform_reads(fp, 1);
			adios_selection_delete(sel);
			
			for(int i = 0; i < nt; i++) 
			{
				arr->SetValue(idx, (int)buff[i]);
				idx++;
			}
			delete [] buff;
		}	
		field = new eavlField(1, arr, eavlField::ASSOC_POINTS);
	}
	else if((name.compare("egid") == 0)) 
	{
		eavlIntArray *arr = new eavlIntArray(name, 1);
		arr->SetNumberOfTuples(emaxgid);
		for (it = egid.begin(); it != egid.end(); it++) 
		{
			int nt = 1;
			ADIOS_SELECTION *sel = MakeSelection(it->second, s, c);
			for(int i = 0; i < it->second->ndim; i++)
				nt *= c[i];
			long long *buff = new long long[nt];
			adios_schedule_read_byid(fp, sel, it->second->varid, 0, 1, buff);
			int retval = adios_perform_reads(fp, 1);
			adios_selection_delete(sel);

			for(int i = 0; i < nt; i++)
			{
				arr->SetValue(idx, (int)buff[i]);
				idx++;
			}
			delete [] buff;
		}
		field = new eavlField(1, arr, eavlField::ASSOC_POINTS);
	} 
	else 
	{
		THROW(eavlException, string("Variable not found: ")+name);
	}
    return field;
}


ADIOS_SELECTION *
eavlXGCParticleImporter::MakeSelection(ADIOS_VARINFO *avi, uint64_t *s, uint64_t *c)
{
    for(int i = 0; i < 3; i++)
		s[i] = c[i] = 0;
    for(int i = 0; i < avi->ndim; i++)
		c[i] = avi->dims[i];
    
    return adios_selection_boundingbox(avi->ndim, s, c);
}

