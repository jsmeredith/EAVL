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
    totalIParticles = 0;
    totalEParticles = 0;
    iphaseAvail = 0;
    ephaseAvail = 0;
    fp = NULL;
    getR = true;
    getZ = true;
    getPhi = true;
    getRho = true;
    getW1 = true;
    getW2 = true;
    getMu = true;
    getW0 = true;
    getF0 = true;
    getOriginNode = true;


    std::string key (".restart");
    std::size_t found = filename.find(key);
    if(found != std::string::npos)
        readingRestartFile = 1;
    else
        readingRestartFile = 0;


    MPI_Comm comm_dummy = comm = 0;
    char    hostname[MPI_MAX_PROCESSOR_NAME];
    char    str [256];
    int     len = 0;

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
eavlXGCParticleImporter::eavlXGCParticleImporter(   const string &filename,
                                                    ADIOS_READ_METHOD method,
                                                    MPI_Comm communicator,
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
    retVal = 0;
    totalIParticles = 0;
    totalEParticles = 0;
    fp = NULL;
    fp = NULL;
    getR = true;
    getZ = true;
    getPhi = true;
    getRho = true;
    getW1 = true;
    getW2 = true;
    getMu = true;
    getW0 = true;
    getF0 = true;
    getOriginNode = true;
    comm = communicator;


    std::string key (".restart");
    std::size_t found = filename.find(key);
    if(found != std::string::npos)
        readingRestartFile = 1;
    else
        readingRestartFile = 0;


    char    hostname[MPI_MAX_PROCESSOR_NAME];
    char    str [256];
    int     len = 0;

    //Set local mpi vars so we know how many minions there are, and wich we are
    MPI_Comm_size(comm,&numMPITasks);
    MPI_Comm_rank(comm,&mpiRank);
    MPI_Get_processor_name(hostname, &len);

    fp = adios_read_open(filename.c_str(), method, comm, mode, timeout_sec);

    if (fp == NULL)
    {
        if(adios_errno == err_end_of_stream)
        {
            printf ("End of stream, no more steps expected. Quit. %s\n",
                        adios_errmsg()
                   );
        }
        else
        {
            printf ("No new step arrived within the timeout. Quit. %s\n",
                     adios_errmsg()
                    );
            THROW(eavlException, "XGC variable file not found.");
        }
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

#define AdiosGetValue(fp, varid, data) \
                adios_schedule_read_byid (fp, 0, varid, fp->current_step, 1, &data); \
                adios_perform_reads (fp, 1);


void
eavlXGCParticleImporter::Initialize()
{
    ephase.clear();
    iphase.clear();
    egid.clear();
    igid.clear();

    last_available_timestep = fp->last_step;

    if(readingRestartFile)
    {
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

            if(varNm == "ephase")
            {
                ephase[longvarNm] = avi;
                ephaseAvail = 1;
            }
            else if(varNm == "egid")
            {
                egid[longvarNm] = avi;
            }
            else if(varNm == "iphase")
            {
                iphase[longvarNm] = avi;
                iphaseAvail = 1;
            }
            else if(varNm == "igid")
            {
                igid[longvarNm] = avi;
            }
            else if(varNm == "inum")
            {
                int ipart;
                AdiosGetValue (fp, i, ipart);
                totalIParticles += ipart;
                //totalIParticles += (int)(*(int *)avi->value);
                adios_free_varinfo(avi);
            }
            else if(varNm == "enum")
            {
                int epart;
                AdiosGetValue (fp, i, epart);
                totalEParticles += epart;
                //totalEParticles += (int)(*(int *)avi->value);
                adios_free_varinfo(avi);
            }
            else if(i < startIndex + 13)
            {
                if (varNm == "timestep")
                {
                    //timestep = (int)(*(int *)avi->value);
                    AdiosGetValue (fp, i, timestep);
                }
                else if(varNm == "time")
                {
                    AdiosGetValue (fp, i, time);
                    //time = (double)(*(double *)avi->value);
                }
                else if(varNm == "maxnum")
                {
                    AdiosGetValue (fp, i, maxnum);
                    //maxnum = (int)(*(int *)avi->value);
                }
                else if (varNm == "inphase")
                {
                    AdiosGetValue (fp, i, inphase);
                    //inphase = (int)(*(int *)avi->value);
                }
                else if(varNm == "enphase")
                {
                    AdiosGetValue (fp, i, enphase);
                    //enphase = (int)(*(int *)avi->value);
                }
                else if(varNm == "emaxgid")
                {
                    AdiosGetValue (fp, i, emaxgid);
                    //emaxgid = (long long)(*(long long *)avi->value);
                }
                else if(varNm == "imaxgid")
                {
                    AdiosGetValue (fp, i, imaxgid);
                    //imaxgid = (long long)(*(long long *)avi->value);
                }
                adios_free_varinfo(avi);
            }
            else
            {
                adios_free_varinfo(avi);
            }
        }
    }
    else
    {

        for(int i = 0; i < fp->nvars; i++)
        {
            string varNm(&fp->var_namelist[i][0]);

            if(varNm == "ephase")
            {
                ephaseAvail = 1;
                //----Set indexes for each reader for the ELECTRONS
                ADIOS_VARINFO *avi = adios_inq_var(fp, "ephase");
                nvars = avi->dims[0];
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
                ELECTRONendIndex = endIndex;
                ELECTRONstartIndex = startIndex;
                adios_free_varinfo(avi);
                //--
            }
            else if(varNm == "iphase")
            {
                iphaseAvail = 1;
                //----Set indexes for each reader for the IONS
                ADIOS_VARINFO *avi = adios_inq_var(fp, "iphase");
                nvars = avi->dims[0];
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
                IONendIndex = endIndex;
                IONstartIndex = startIndex;
                adios_free_varinfo(avi);
                //--
            }
        }

        for(int i = 0; i < fp->nvars; i++)
        {
            ADIOS_VARINFO *avi = adios_inq_var_byid(fp, i);
            string varNm(&fp->var_namelist[i][0]);

            if(varNm == "ephase")
            {
                ephase[varNm] = avi;
            }
            else if(varNm == "egid")
            {
                egid[varNm] = avi;
            }
            else if(varNm == "iphase")
            {
                iphase[varNm] = avi;
            }
            else if(varNm == "igid")
            {
                igid[varNm] = avi;
            }
            else if(varNm == "inum")
            {
                totalIParticles = IONendIndex - IONstartIndex;
                adios_free_varinfo(avi);
            }
            else if(varNm == "enum")
            {
                totalEParticles = ELECTRONendIndex - ELECTRONstartIndex;
                adios_free_varinfo(avi);
            }
            else if (varNm == "timestep")
            {
                //timestep = (int)(*(int *)avi->value);
                AdiosGetValue (fp, i, timestep);
                adios_free_varinfo(avi);
            }
            else if(varNm == "time")
            {
                AdiosGetValue (fp, i, time);
                //time = (double)(*(double *)avi->value);
                adios_free_varinfo(avi);
            }
            else if (varNm == "inphase")
            {
                AdiosGetValue (fp, i, inphase);
                //inphase = (int)(*(int *)avi->value);
                adios_free_varinfo(avi);
            }
            else if(varNm == "enphase")
            {
                AdiosGetValue (fp, i, enphase);
                //enphase = (int)(*(int *)avi->value);
                adios_free_varinfo(avi);
            }
            else if(varNm == "enum_total")
            {
                AdiosGetValue (fp, i, emaxgid);
                //emaxgid = (long long)(*(long long *)avi->value);
                adios_free_varinfo(avi);
            }
            else if(varNm == "inum_total")
            {
                AdiosGetValue (fp, i, imaxgid);
                //imaxgid = (long long)(*(long long *)avi->value);
                adios_free_varinfo(avi);
            }
            else
            {
                 adios_free_varinfo(avi);
            }
        }
    }
}

vector<string>
eavlXGCParticleImporter::GetMeshList()
{
    vector<string> m;
    if(iphaseAvail) m.push_back("iMesh");
    if(ephaseAvail) m.push_back("eMesh");
    return m;
}

vector<string>
eavlXGCParticleImporter::GetCellSetList(const std::string &mesh)
{
    vector<string> m;
    if(mesh == "iMesh" && iphaseAvail)
        m.push_back("iMesh_cells");
    else if(mesh == "eMesh" && ephaseAvail)
        m.push_back("eMesh_cells");
    return m;
}

vector<string>
eavlXGCParticleImporter::GetFieldList(const std::string &mesh)
{
    vector<string> fields;
    if(mesh == "iMesh" && iphaseAvail)
    {
        fields.push_back("iphase");
        fields.push_back("igid");
    }
    else if(mesh == "eMesh" && ephaseAvail)
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

    if(name == "iMesh" && iphaseAvail)
    {
        ds->SetNumPoints(totalIParticles);
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
        axisValues[0]->SetNumberOfTuples(totalIParticles);
        axisValues[1]->SetNumberOfTuples(totalIParticles);
        axisValues[2]->SetNumberOfTuples(totalIParticles);

        //Set all of the axis values to the x, y, z coordinates of the
        //iphase particles; set computational node origin
        eavlIntArray *originNode;
        if(getOriginNode && readingRestartFile)
            originNode = new eavlIntArray("originNode", 1, totalIParticles);
        eavlFloatArray *r;
        if(getR)
            r = new eavlFloatArray("R", 1, totalIParticles);
        eavlFloatArray *z;
        if(getZ)
            z = new eavlFloatArray("Z", 1, totalIParticles);
        eavlFloatArray *phi;
        if(getPhi)
            phi = new eavlFloatArray("phi", 1, totalIParticles);
        eavlFloatArray *rho;
        if(getRho)
            rho = new eavlFloatArray("rho", 1, totalIParticles);
        eavlFloatArray *w1;
        if(getW1)
            w1 = new eavlFloatArray("w1", 1, totalIParticles);
        eavlFloatArray *w2;
        if(getW2)
            w2 = new eavlFloatArray("w2", 1, totalIParticles);
        eavlFloatArray *mu;
        if(getMu)
            mu = new eavlFloatArray("mu", 1, totalIParticles);
        eavlFloatArray *w0;
        if(getW0)
            w0 = new eavlFloatArray("w0", 1, totalIParticles);
        eavlFloatArray *f0;
        if(getF0)
            f0 = new eavlFloatArray("f0", 1, totalIParticles);

        uint64_t s[3], c[3];
        double *buff;
        int nt = 1, idx = 0;
        map<string, ADIOS_VARINFO*>::const_iterator it;
        for(it = iphase.begin(); it != iphase.end(); it++)
        {
            ADIOS_SELECTION *sel;
            if(readingRestartFile)
                sel = MakeSelection(it->second, s, c);
            else
                sel = MakeLimitedSelection(it->second, s, c, 1);

            nt = 1;
            for (int i = 0; i < it->second->ndim; i++)
                nt *= c[i];

            buff = new double[nt];
            adios_schedule_read_byid(fp, sel, it->second->varid, 0, 1, buff);
            adios_perform_reads(fp, 1);
            adios_selection_delete(sel);

            string nodeNum;
            if(getOriginNode && readingRestartFile) nodeNum = it->first.substr(it->first.find("_",1,1)+1, 5);

            for(int i = 0; i < nt; i+=9)
            {
                if(getR)   r->SetValue(idx, buff[i]);
                if(getZ)   z->SetValue(idx, buff[i+1]);
                if(getPhi) phi->SetValue(idx, buff[i+2]);
                if(getRho) rho->SetValue(idx, buff[i+3]);
                if(getW1)  w1->SetValue(idx, buff[i+4]);
                if(getW2)  w2->SetValue(idx, buff[i+5]);
                if(getMu)  mu->SetValue(idx, buff[i+6]);
                if(getW0)  w0->SetValue(idx, buff[i+7]);
                if(getF0)  f0->SetValue(idx, buff[i+8]);
                axisValues[0]->SetComponentFromDouble(idx, 0, r->GetValue(idx)*cos(phi->GetValue(idx)));
                axisValues[1]->SetComponentFromDouble(idx, 0, r->GetValue(idx)*sin(phi->GetValue(idx)));
                axisValues[2]->SetComponentFromDouble(idx, 0, z->GetValue(idx));
                if(getOriginNode && readingRestartFile) originNode->SetValue(idx, atoi(nodeNum.c_str()));
                idx++;
            }
            delete [] buff;
        }

        ds->AddField(new eavlField(1, axisValues[0], eavlField::ASSOC_POINTS));
        ds->AddField(new eavlField(1, axisValues[1], eavlField::ASSOC_POINTS));
        ds->AddField(new eavlField(1, axisValues[2], eavlField::ASSOC_POINTS));
        if(getOriginNode && readingRestartFile)
            ds->AddField(new eavlField(1, originNode, eavlField::ASSOC_POINTS));
        if(getR)   ds->AddField(new eavlField(1, r,   eavlField::ASSOC_POINTS));
        if(getZ)   ds->AddField(new eavlField(1, z,   eavlField::ASSOC_POINTS));
        if(getPhi) ds->AddField(new eavlField(1, phi, eavlField::ASSOC_POINTS));
        if(getRho) ds->AddField(new eavlField(1, rho, eavlField::ASSOC_POINTS));
        if(getW1)  ds->AddField(new eavlField(1, w1,  eavlField::ASSOC_POINTS));
        if(getW2)  ds->AddField(new eavlField(1, w2,  eavlField::ASSOC_POINTS));
        if(getMu)  ds->AddField(new eavlField(1, mu,  eavlField::ASSOC_POINTS));
        if(getW0)  ds->AddField(new eavlField(1, w0,  eavlField::ASSOC_POINTS));
        if(getF0)  ds->AddField(new eavlField(1, f0,  eavlField::ASSOC_POINTS));

        eavlCellSet *cellSet = new eavlCellSetAllPoints(name + "_cells", totalIParticles);
        ds->AddCellSet(cellSet);
        //-- END set axis values

        //----Set the ids of all axis values
        idx = 0;
        long long *idBuff;
        eavlIntArray *axisIds = new eavlIntArray("id", 1, totalIParticles);
        for(it = igid.begin(); it != igid.end(); it++)
        {
            ADIOS_SELECTION *sel;
            if(readingRestartFile)
                sel = MakeSelection(it->second, s, c);
            else
                sel = MakeLimitedSelection(it->second, s, c, 1);

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
    else if(name == "eMesh" && ephaseAvail)
    {
        ds->SetNumPoints(totalEParticles);
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
        axisValues[0]->SetNumberOfTuples(totalEParticles);
        axisValues[1]->SetNumberOfTuples(totalEParticles);
        axisValues[2]->SetNumberOfTuples(totalEParticles);

        //Set all of the axis values to the x, y, z coordinates of the
        //ephase particles; set computational node origin
        eavlIntArray *originNode;
        if(getOriginNode && readingRestartFile)
            originNode = new eavlIntArray("originNode", 1, totalIParticles);
        eavlFloatArray *r;
        if(getR)
            r = new eavlFloatArray("R", 1, totalIParticles);
        eavlFloatArray *z;
        if(getZ)
            z = new eavlFloatArray("Z", 1, totalIParticles);
        eavlFloatArray *phi;
        if(getPhi)
            phi = new eavlFloatArray("phi", 1, totalIParticles);
        eavlFloatArray *rho;
        if(getRho)
            rho = new eavlFloatArray("rho", 1, totalIParticles);
        eavlFloatArray *w1;
        if(getW1)
            w1 = new eavlFloatArray("w1", 1, totalIParticles);
        eavlFloatArray *w2;
        if(getW2)
            w2 = new eavlFloatArray("w2", 1, totalIParticles);
        eavlFloatArray *mu;
        if(getMu)
            mu = new eavlFloatArray("mu", 1, totalIParticles);
        eavlFloatArray *w0;
        if(getW0)
            w0 = new eavlFloatArray("w0", 1, totalIParticles);
        eavlFloatArray *f0;
        if(getF0)
            f0 = new eavlFloatArray("f0", 1, totalIParticles);

        uint64_t s[3], c[3];
        int nt = 1, idx = 0;
        double *buff;
        map<string, ADIOS_VARINFO*>::const_iterator it;
        for(it = ephase.begin(); it != ephase.end(); it++)
        {
            ADIOS_SELECTION *sel;
            if(readingRestartFile)
                sel = MakeSelection(it->second, s, c);
            else
                sel = MakeLimitedSelection(it->second, s, c, 0);

            nt = 1;
            for(int i = 0; i < it->second->ndim; i++)
                nt *= c[i];

            buff = new double[nt];
            adios_schedule_read_byid(fp, sel, it->second->varid, 0, 1, buff);
            adios_perform_reads(fp, 1);
            adios_selection_delete(sel);

            string nodeNum;
            if(getOriginNode && readingRestartFile) nodeNum = it->first.substr(it->first.find("_",1,1)+1, 5);

            for(int i = 0; i < nt; i+=9)
            {
                if(getR)   r->SetValue(idx, buff[i]);
                if(getZ)   z->SetValue(idx, buff[i+1]);
                if(getPhi) phi->SetValue(idx, buff[i+2]);
                if(getRho) rho->SetValue(idx, buff[i+3]);
                if(getW1)  w1->SetValue(idx, buff[i+4]);
                if(getW2)  w2->SetValue(idx, buff[i+5]);
                if(getMu)  mu->SetValue(idx, buff[i+6]);
                if(getW0)  w0->SetValue(idx, buff[i+7]);
                if(getF0)  f0->SetValue(idx, buff[i+8]);
                axisValues[0]->SetComponentFromDouble(idx, 0, r->GetValue(idx)*cos(phi->GetValue(idx)));
                axisValues[1]->SetComponentFromDouble(idx, 0, r->GetValue(idx)*sin(phi->GetValue(idx)));
                axisValues[2]->SetComponentFromDouble(idx, 0, z->GetValue(idx));
                if(getOriginNode && readingRestartFile) originNode->SetValue(idx, atoi(nodeNum.c_str()));
                idx++;
            }
            delete [] buff;
        }

        ds->AddField(new eavlField(1, axisValues[0], eavlField::ASSOC_POINTS));
        ds->AddField(new eavlField(1, axisValues[1], eavlField::ASSOC_POINTS));
        ds->AddField(new eavlField(1, axisValues[2], eavlField::ASSOC_POINTS));
        if(getOriginNode && readingRestartFile)
            ds->AddField(new eavlField(1, originNode, eavlField::ASSOC_POINTS));
        if(getR)   ds->AddField(new eavlField(1, r,   eavlField::ASSOC_POINTS));
        if(getZ)   ds->AddField(new eavlField(1, z,   eavlField::ASSOC_POINTS));
        if(getPhi) ds->AddField(new eavlField(1, phi, eavlField::ASSOC_POINTS));
        if(getRho) ds->AddField(new eavlField(1, rho, eavlField::ASSOC_POINTS));
        if(getW1)  ds->AddField(new eavlField(1, w1,  eavlField::ASSOC_POINTS));
        if(getW2)  ds->AddField(new eavlField(1, w2,  eavlField::ASSOC_POINTS));
        if(getMu)  ds->AddField(new eavlField(1, mu,  eavlField::ASSOC_POINTS));
        if(getW0)  ds->AddField(new eavlField(1, w0,  eavlField::ASSOC_POINTS));
        if(getF0)  ds->AddField(new eavlField(1, f0,  eavlField::ASSOC_POINTS));

        eavlCellSet *cellSet = new eavlCellSetAllPoints(name + "_cells", totalEParticles);
        ds->AddCellSet(cellSet);
        //-- END set axis values


        //----Set the ids of all axis values
        idx = 0;
        long long *idBuff;
        eavlIntArray *axisIds = new eavlIntArray("id", 1, totalEParticles);

        for(it = egid.begin(); it != egid.end(); it++)
        {
            ADIOS_SELECTION *sel;
            if(readingRestartFile)
                sel = MakeSelection(it->second, s, c);
            else
                sel = MakeLimitedSelection(it->second, s, c, 0);

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
    if(name.compare("ephase") == 0 && ephaseAvail)
    {
        eavlFloatArray *arr = new eavlFloatArray(name, 1);
        arr->SetNumberOfTuples(totalEParticles*9);
        for(it = ephase.begin(); it != ephase.end(); it++)
        {
            int nt = 1;
            ADIOS_SELECTION *sel;
            if(readingRestartFile)
                sel = MakeSelection(it->second, s, c);
            else
                sel = MakeLimitedSelection(it->second, s, c, 0);

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
    else if(name.compare("iphase") == 0 && iphaseAvail)
    {
        eavlFloatArray *arr = new eavlFloatArray(name, 1);
        arr->SetNumberOfTuples(totalIParticles*9);
        for(it = iphase.begin(); it != iphase.end(); it++)
        {
            int nt = 1;
            ADIOS_SELECTION *sel;
            if(readingRestartFile)
                sel = MakeSelection(it->second, s, c);
            else
                sel = MakeLimitedSelection(it->second, s, c, 1);

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
    else if(name.compare("igid") == 0 && iphaseAvail)
    {
        eavlIntArray *arr = new eavlIntArray(name, 1);
        arr->SetNumberOfTuples(totalIParticles);
        for(it = igid.begin(); it != igid.end(); it++)
        {
            int nt = 1;
            ADIOS_SELECTION *sel;
            if(readingRestartFile)
                sel = MakeSelection(it->second, s, c);
            else
                sel = MakeLimitedSelection(it->second, s, c, 1);

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
    else if((name.compare("egid") == 0) && ephaseAvail )
    {
        eavlIntArray *arr = new eavlIntArray(name, 1);
        arr->SetNumberOfTuples(totalEParticles);
        for (it = egid.begin(); it != egid.end(); it++)
        {
            int nt = 1;
            ADIOS_SELECTION *sel;
            if(readingRestartFile)
                sel = MakeSelection(it->second, s, c);
            else
                sel = MakeLimitedSelection(it->second, s, c, 0);

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
eavlXGCParticleImporter::MakeLimitedSelection(ADIOS_VARINFO *avi, uint64_t *s, uint64_t *c, int ION)
{
    if(ION)
    {
        s[0] = IONstartIndex;
        s[1] = 0;
        s[2] = 0;
        c[0] = IONendIndex-IONstartIndex;
        c[1] = 0;
        c[2] = 0;

        for(int i = 1; i < avi->ndim; i++)
        {
            c[i] = avi->dims[i];
        }
    }
    else
    {
        s[0] = ELECTRONstartIndex;
        s[1] = 0;
        s[2] = 0;
        c[0] = ELECTRONendIndex - ELECTRONstartIndex;
        c[1] = 0;
        c[2] = 0;

        for(int i = 1; i < avi->ndim; i++)
        {
            c[i] = avi->dims[i];
        }
    }

    return adios_selection_boundingbox(avi->ndim, s, c);
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
eavlXGCParticleImporter::GetEMaxGID()
{
    return imaxgid;
}

int
eavlXGCParticleImporter::GetIMaxGID()
{
    return emaxgid;
}

int
eavlXGCParticleImporter::GetLastTimeStep()
{
    return last_available_timestep;
}

int
eavlXGCParticleImporter::AdvanceToTimeStep(int step, int timeout_sec)
{
    int currentTimestep = timestep;
    while(currentTimestep < step)
    {
        int err = adios_advance_step(fp, 0, timeout_sec);

        if(err != 0)
            return err;
        currentTimestep++;
    }

    timestep = 0;
    maxnum = 0;
    enphase = 0;
    inphase = 0;
    emaxgid = 0;
    imaxgid = 0;
    nvars = 0;
    time = 0;
    retVal = 0;
    totalIParticles = 0;
    totalEParticles = 0;

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

int
eavlXGCParticleImporter::AdvanceTimeStep(int step, int timeout_sec)
{
    int err = adios_advance_step(fp, step, timeout_sec);

    if(err != 0)
        return err;

    timestep = 0;
    maxnum = 0;
    enphase = 0;
    inphase = 0;
    emaxgid = 0;
    imaxgid = 0;
    nvars = 0;
    time = 0;
    retVal = 0;
    totalIParticles = 0;
    totalEParticles = 0;

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

void
eavlXGCParticleImporter::SetActiveFields(bool r, bool z, bool phi, bool rho, bool w1, bool w2,
                                         bool mu, bool w0, bool f0, bool originNode
                                        )
{
    getR = r;
    getZ = z;
    getPhi = phi;
    getRho = rho;
    getW1 = w1;
    getW2 = w2;
    getMu = mu;
    getW0 = w0;
    getF0 = f0;
    getOriginNode = originNode;
}
