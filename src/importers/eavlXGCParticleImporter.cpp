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
    retVal = 0;
    fp = NULL;
    
    string::size_type i0 = filename.rfind("xgc.");
    string::size_type i1 = filename.rfind(".bp");

    MPI_Comm comm_dummy = comm = 0;
    
    char 		hostname[MPI_MAX_PROCESSOR_NAME];
    char        str [256];
    int			len = 0;
    
    //Set local mpi vars so we know how many minions there are, and wich we are
    MPI_Comm_size(comm,&numMPITasks);
	MPI_Comm_rank(comm,&mpiRank);
	MPI_Get_processor_name(hostname, &len);
	
    fp = adios_read_open_file(filename.c_str(), ADIOS_READ_METHOD_BP, comm_dummy);
    
    if(fp == NULL)
		THROW(eavlException, "XGC variable file not found.");

    Initialize();
}


//Reads an adios restart file :: can only be instantiated if EAVL was built
//with MPI enabled!
eavlXGCParticleImporter::eavlXGCParticleImporter(	const string &filename, 
													ADIOS_READ_METHOD method, 
													MPI_Comm communicator, 
													ADIOS_LOCKMODE mode, 
													int timeout_sec,
													int fromDataspaces
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
    retVal = 0;
    fp = NULL;
								
    string::size_type i0 = filename.rfind("xgc.");
    string::size_type i1 = filename.rfind(".bp");
    comm = communicator;
    
    char 		hostname[MPI_MAX_PROCESSOR_NAME];
    char        str [256];
    int len = 0;
    
    //Set local mpi vars so we know how many minions there are, and wich we are
    MPI_Comm_size(comm,&numMPITasks);
	MPI_Comm_rank(comm,&mpiRank);
	MPI_Get_processor_name(hostname, &len);
    
    if(fromDataspaces)
	    fp = adios_read_open(filename.c_str(), method, comm, mode, timeout_sec);
    else
    	fp = adios_read_open_file(filename.c_str(), method, comm);
    	
    if(fp == NULL)
    {
    	fprintf (stderr, "Error at opening adios stream: %s\n", adios_errmsg());
		THROW(eavlException, "Adios stream error, or XGC variable file not found.");
	}
	
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
    
    nvars = fp->nvars/13;
    
    if(nvars <= mpiRank)
    {
    	printf("Warning! :: Thread[%i] is wasting cycles :: too many processors for data\n", mpiRank);
    	return;    	
    }
    
    //----Set indexes for each reader if there is more than one
    int endIndex;
    int startIndex = (nvars / numMPITasks) * mpiRank;
  	if (nvars % numMPITasks > mpiRank)
  	{
    	startIndex += mpiRank;
    	endIndex = startIndex + (nvars / numMPITasks) + 1;
  	}
  	else
  	{
    	startIndex += nvars % numMPITasks;
    	endIndex = startIndex + (nvars / numMPITasks);
  	}
	startIndex *= 13;
	endIndex *= 13;
	//--
	
    for(int i = startIndex; i < endIndex; i++)
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
		else if(i < startIndex + 13) 
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
		eavlFloatArray *r = new eavlFloatArray("R", 1, imaxgid);
		eavlFloatArray *z = new eavlFloatArray("Z", 1, imaxgid);
		eavlFloatArray *phi = new eavlFloatArray("phi", 1, imaxgid);
		eavlFloatArray *rho = new eavlFloatArray("rho", 1, imaxgid);
		eavlFloatArray *w1 = new eavlFloatArray("w1", 1, imaxgid);
		eavlFloatArray *w2 = new eavlFloatArray("w2", 1, imaxgid);
		eavlFloatArray *mu = new eavlFloatArray("mu", 1, imaxgid);
		eavlFloatArray *w0 = new eavlFloatArray("w0", 1, imaxgid);
		eavlFloatArray *f0 = new eavlFloatArray("f0", 1, imaxgid);
		
		uint64_t s[3], c[3];
		double *buff;
		int nt = 1, idx = 0;
		map<string, ADIOS_VARINFO*>::const_iterator it;
		for(it = iphase.begin(); it != iphase.end(); it++) 
		{
			ADIOS_SELECTION *sel = MakeSelection(it->second, s, c);
			nt = 1;
			for (int i = 0; i < it->second->ndim; i++)
				nt *= c[i];
			
			buff = new double[nt];
			adios_schedule_read_byid(fp, sel, it->second->varid, 0, 1, buff);
			adios_perform_reads(fp, 1);
			adios_selection_delete(sel);
			
			string nodeNum = it->first.substr(it->first.find("_",1,1)+1, 5);
			
			for(int i = 0; i < nt; i+=9) 
			{
				r->SetValue(idx, buff[i]);
				z->SetValue(idx, buff[i+1]);
				phi->SetValue(idx, buff[i+2]);
				rho->SetValue(idx, buff[i+3]);
				w1->SetValue(idx, buff[i+4]);
				w2->SetValue(idx, buff[i+5]);
				mu->SetValue(idx, buff[i+6]);
				w0->SetValue(idx, buff[i+7]);
				f0->SetValue(idx, buff[i+8]);
				axisValues[0]->SetComponentFromDouble(idx, 0, r->GetValue(idx)*cos(phi->GetValue(idx)));
				axisValues[1]->SetComponentFromDouble(idx, 0, r->GetValue(idx)*sin(phi->GetValue(idx)));
				axisValues[2]->SetComponentFromDouble(idx, 0, z->GetValue(idx));
				originNode->SetValue(idx, atoi(nodeNum.c_str()));
				idx++;
			}
			delete [] buff;
		}
		
		ds->AddField(new eavlField(1, axisValues[0], eavlField::ASSOC_POINTS));
		ds->AddField(new eavlField(1, axisValues[1], eavlField::ASSOC_POINTS));
		ds->AddField(new eavlField(1, axisValues[2], eavlField::ASSOC_POINTS));
		ds->AddField(new eavlField(1, originNode, eavlField::ASSOC_POINTS));
		ds->AddField(new eavlField(1, r, eavlField::ASSOC_POINTS));
		ds->AddField(new eavlField(1, z, eavlField::ASSOC_POINTS));
		ds->AddField(new eavlField(1, phi, eavlField::ASSOC_POINTS));
		ds->AddField(new eavlField(1, rho, eavlField::ASSOC_POINTS));
		ds->AddField(new eavlField(1, w1, eavlField::ASSOC_POINTS));
		ds->AddField(new eavlField(1, w2, eavlField::ASSOC_POINTS));
		ds->AddField(new eavlField(1, mu, eavlField::ASSOC_POINTS));
		ds->AddField(new eavlField(1, w0, eavlField::ASSOC_POINTS));
		ds->AddField(new eavlField(1, f0, eavlField::ASSOC_POINTS));
		
		eavlCellSet *cellSet = new eavlCellSetAllPoints(name + "_cells", imaxgid);
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
			adios_perform_reads(fp, 1);
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
		//ephase particles; set computational node origin
		eavlIntArray *originNode = new eavlIntArray("originNode", 1, emaxgid);
		eavlFloatArray *r = new eavlFloatArray("R", 1, imaxgid);
		eavlFloatArray *z = new eavlFloatArray("Z", 1, imaxgid);
		eavlFloatArray *phi = new eavlFloatArray("phi", 1, imaxgid);
		eavlFloatArray *rho = new eavlFloatArray("rho", 1, imaxgid);
		eavlFloatArray *w1 = new eavlFloatArray("w1", 1, imaxgid);
		eavlFloatArray *w2 = new eavlFloatArray("w2", 1, imaxgid);
		eavlFloatArray *mu = new eavlFloatArray("mu", 1, imaxgid);
		eavlFloatArray *w0 = new eavlFloatArray("w0", 1, imaxgid);
		eavlFloatArray *f0 = new eavlFloatArray("f0", 1, imaxgid);
		
		uint64_t s[3], c[3];
		int nt = 1, idx = 0;
		double *buff;
		map<string, ADIOS_VARINFO*>::const_iterator it;
		for(it = ephase.begin(); it != ephase.end(); it++) 
		{
			ADIOS_SELECTION *sel = MakeSelection(it->second, s, c);
			nt = 1;
			for(int i = 0; i < it->second->ndim; i++)
				nt *= c[i];
			
			buff = new double[nt];
			adios_schedule_read_byid(fp, sel, it->second->varid, 0, 1, buff);
			adios_perform_reads(fp, 1);
			adios_selection_delete(sel);
			
			string nodeNum = it->first.substr(it->first.find("_",1,1)+1, 5);
			
			for(int i = 0; i < nt; i+=9) 
			{
				r->SetValue(idx, buff[i]);
				z->SetValue(idx, buff[i+1]);
				phi->SetValue(idx, buff[i+2]);
				rho->SetValue(idx, buff[i+3]);
				w1->SetValue(idx, buff[i+4]);
				w2->SetValue(idx, buff[i+5]);
				mu->SetValue(idx, buff[i+6]);
				w0->SetValue(idx, buff[i+7]);
				f0->SetValue(idx, buff[i+8]);
				axisValues[0]->SetComponentFromDouble(idx, 0, r->GetValue(idx)*cos(phi->GetValue(idx)));
				axisValues[1]->SetComponentFromDouble(idx, 0, r->GetValue(idx)*sin(phi->GetValue(idx)));
				axisValues[2]->SetComponentFromDouble(idx, 0, z->GetValue(idx));
				originNode->SetValue(idx, atoi(nodeNum.c_str()));
				idx++;
			}
			delete [] buff;
		}
		
		ds->AddField(new eavlField(1, axisValues[0], eavlField::ASSOC_POINTS));
		ds->AddField(new eavlField(1, axisValues[1], eavlField::ASSOC_POINTS));
		ds->AddField(new eavlField(1, axisValues[2], eavlField::ASSOC_POINTS));
		ds->AddField(new eavlField(1, originNode, eavlField::ASSOC_POINTS));
		ds->AddField(new eavlField(1, r, eavlField::ASSOC_POINTS));
		ds->AddField(new eavlField(1, z, eavlField::ASSOC_POINTS));
		ds->AddField(new eavlField(1, phi, eavlField::ASSOC_POINTS));
		ds->AddField(new eavlField(1, rho, eavlField::ASSOC_POINTS));
		ds->AddField(new eavlField(1, w1, eavlField::ASSOC_POINTS));
		ds->AddField(new eavlField(1, w2, eavlField::ASSOC_POINTS));
		ds->AddField(new eavlField(1, mu, eavlField::ASSOC_POINTS));
		ds->AddField(new eavlField(1, w0, eavlField::ASSOC_POINTS));
		ds->AddField(new eavlField(1, f0, eavlField::ASSOC_POINTS));
		
		eavlCellSet *cellSet = new eavlCellSetAllPoints(name + "_cells", emaxgid);
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
			adios_perform_reads(fp, 1);
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

int
eavlXGCParticleImporter::GetTimeStep()
{
	return timestep;
}

int
eavlXGCParticleImporter::AdvanceTimeStep(int step, int timeout_sec)
{
	int err = adios_advance_step (fp, step, timeout_sec);

	if(err != 0)
		return -1;				 
	
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
	
	Initialize();
	return 0;
}
