#! /usr/bin/env python

# *****************************************************************************
# File: runtest.py
#
# Purpose:
#    Runs a set of tests against sample data files and validates the
#    results against known baselines.  For now, the results are simply
#    a summary of the generated data set (showing the overall structure
#    and a small set of values for each field).
#
# Programmer:  Jeremy Meredith
# Creation:    July 23, 2012
#
# Modifications:
# *****************************************************************************

# -----------------------------------------------------------------------------
#                              Infrastructure
# -----------------------------------------------------------------------------

import subprocess, os, sys

def AddResult(category, exitcode, coutfn, cerrfn):
    runcounts.setdefault(category, 0)
    runcounts[category] += 1
    
    successcounts.setdefault(category, 0)
    errorcounts.setdefault(category, 0)

    basefile = "baseline/"+coutfn
    currfile = "current/"+coutfn

    if not os.path.exists(basefile):
        # missing baseline result; shouldn't happen
        logfile.write("%s: Missing baseline result\n" % coutfn)
        rebasefile.write("cp %s %s\n" % (currfile, basefile))
    elif not os.path.exists(currfile):
        # missing current result; shouldn't happen
        logfile.write("%s: Missing current result\n" % coutfn)
    else:
        devnull=open("/dev/null","w")
        diff = subprocess.call(["diff", "-q", basefile, currfile], stdout=devnull, stderr=devnull)
        if exitcode == 0:
            if diff:
                logfile.write("%s: Different result; regression possibly introduced.\n" % coutfn)
                difffile.write("echo; echo \"------------ %s ------------\"\n" % coutfn)
                difffile.write("diff %s %s\n" % (basefile, currfile))
                tkdifffile.write("echo; echo \"------------ %s ------------\"\n" % coutfn)
                tkdifffile.write("tkdiff %s %s\n" % (basefile, currfile))
                rebasefile.write("cp %s %s\n" % (currfile, basefile))
            else:
                successcounts[category] += 1
                #logfile.write("%s: Success\n" % coutfn)
        else:
            errorcounts[category] += 1
            if diff:
                logfile.write("%s: Failure, exit code was %d.  See file %s for details\n" % (coutfn, exitcode, "current/"+cerrfn))
            else:
                logfile.write("%s: Failure: same result but exit code was %d.  See file %s for details\n" % (coutfn, exitcode, "current/"+cerrfn))

def RunTest(category, fn, args):
    bn = os.path.basename(fn)
    coutfn = "%s/%s.out" % (category,bn)
    cerrfn = "%s/%s.err" % (category,bn)
    cout = open("current/" + coutfn, "w")
    cerr = open("current/" + cerrfn, "w")

    logfile.write("%s\n"%args);
    try:
        exitcode = subprocess.call(args, stdout=cout, stderr=cerr)
    except:
        cerr.write("In runtest: error executing: %s\n" % args)
        exitcode = 99

    cout.close()
    cerr.close()
    AddResult(category, exitcode, coutfn, cerrfn)

def PrintResults():
    totalrun = 0
    totalsuccess = 0
    totalerror = 0
    for k in sections:
        run = runcounts[k]
        success = successcounts[k]
        error = errorcounts[k]
        totalrun += run
        totalsuccess += success
        totalerror += error
        if success == run:
            print "Test case   %-14s:   all %3d tests passed" % ("'%s'"%k, run)
        else:
            print "ERROR: Test %-14s: " % ("'%s'"%k),success,"/",run," tests passed"
    if totalsuccess < totalrun:
        print "\nTesting failure:",totalsuccess,"/",totalrun," tests passed overall"
        if totalerror != 0:
            print "Additionally:",totalerror,"of these were serious failures with non-zero exit code."
        print "\nSee results.txt for more information on failures."
        print "\nRun diffs.sh or tkdiffs.sh to see detailed differences baseline results.\n"
    else:
        if totalerror != 0:
            print "\Failure:",totalerror,"serious failures with non-zero exit code\n"
        else:
            print "\nSUCCESS: all",totalrun," tests passed"
    return totalrun - totalsuccess

def StartSection(category):
    sections.append(category)
    logfile.write("\n===== %s =====\n" % ("%s tests") % category)
    print "Running %s tests" % category


# -----------------------------------------------------------------------------
#                             Test Categories
# -----------------------------------------------------------------------------

#
# Basic file import tests
#
def TestImport(fn):
    RunTest("testimport", fn,
            ["./testimport", fn])

#
# Isosurface tests
#
def TestIso(fn, value, varname):
    RunTest("testiso", fn,
            ["./testiso", "%f"%value, varname, fn])
    # an optional final argument specifies an output file for hand-verification
    #["./testiso", "%f"%value, varname, fn, fn+"-iso.vtk"])

#
# Surface normal tests
#
def TestNormal(fn):
    RunTest("testnormal", fn,
            ["./testnormal", fn])
    # an optional final argument specifies an output file for hand-verification
    #["./testnormal", fn, fn+"-norm.vtk"])

#
# Transform tests
#
def TestTransform(fn):
    RunTest("testxform", fn,
            ["./testxform", fn])
    # an optional final argument specifies an output file for hand-verification
    #["./testxform", fn, fn+"-xform.vtk"])

#
# Distance field tests
#
def TestDistanceField(fn, dims, min, max):
    if (len(dims) == 1):
        RunTest("testdistancefield", fn,
                ["./testdistancefield", fn, "%d"%dims[0], "%f"%min[0], "%f"%max[0]])
    elif (len(dims) == 2):
        RunTest("testdistancefield", fn,
                ["./testdistancefield", fn, "%d"%dims[0], "%d"%dims[1], "%f"%min[0], "%f"%max[0], "%f"%min[1], "%f"%max[1]])
    else: # (dims.length() == 3)
        RunTest("testdistancefield", fn,
                ["./testdistancefield", fn, "%d"%dims[0], "%d"%dims[1], "%d"%dims[2], "%f"%min[0], "%f"%max[0], "%f"%min[1], "%f"%max[1], "%f"%min[2], "%f"%max[2]])
    # an optional final argument specifies an output file for hand-verification

#
# Serialization tests
#
def TestSerialization(fn):
    RunTest("testserialize", fn, ["./testserialize", fn])


# -----------------------------------------------------------------------------
#                               Run Tests
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    runcounts = {}
    successcounts = {}
    errorcounts = {}
    sections = []
    logfile = open("results.txt", "w")
    difffile = open("diffs.sh", "w")
    tkdifffile = open("tkdiffs.sh", "w")
    rebasefile = open("rebaseline.sh", "w")


    StartSection("testimport")
    TestImport("../data/curv_cube.vtk")
    TestImport("../data/curv_x.vtk")
    TestImport("../data/curv_xy.vtk")
    TestImport("../data/curv_xz.vtk")
    TestImport("../data/curv_y.vtk")
    TestImport("../data/curv_yz.vtk")
    TestImport("../data/curv_z.vtk")
    TestImport("../data/poly_1d_in_3d.vtk")
    TestImport("../data/poly_2d_in_3d.vtk")
    TestImport("../data/poly_sphere.vtk")
    TestImport("../data/poly_verts_and_lines.vtk")
    TestImport("../data/poly_verts.vtk")
    TestImport("../data/rect_cube.vtk")
    TestImport("../data/rect_x.vtk")
    TestImport("../data/rect_xy.vtk")
    TestImport("../data/rect_xz.vtk")
    TestImport("../data/rect_y.vtk")
    TestImport("../data/rect_yz.vtk")
    TestImport("../data/rect_z.vtk")
    TestImport("../data/ucd_1d_in_3d.vtk")
    TestImport("../data/ucd_2d_xy.vtk")
    TestImport("../data/ucd_cube.vtk")
    TestImport("../data/ucd_sphere.vtk")
    TestImport("../data/ucd_tets.vtk")

    StartSection("testiso")
    TestIso("../data/rect_cube.vtk", 3.5, "nodal")
    TestIso("../data/curv_cube.vtk", 3.5, "nodal")
    TestIso("../data/ucd_cube.vtk",  3.5, "nodal")
    TestIso("../data/ucd_sphere.vtk",3.5, "nodal")
    TestIso("../data/ucd_tets.vtk",  3.5, "nodal")

    StartSection("testnormal")
    TestNormal("../data/curv_cube.vtk")
    TestNormal("../data/curv_xy.vtk")
    TestNormal("../data/curv_xz.vtk")
    TestNormal("../data/curv_yz.vtk")
    TestNormal("../data/poly_2d_in_3d.vtk")
    TestNormal("../data/poly_sphere.vtk")
    TestNormal("../data/rect_cube.vtk")
    TestNormal("../data/rect_xy.vtk")
    TestNormal("../data/rect_xz.vtk")
    TestNormal("../data/rect_yz.vtk")
    TestNormal("../data/ucd_2d_xy.vtk")
    TestNormal("../data/ucd_cube.vtk")
    TestNormal("../data/ucd_sphere.vtk")
    TestNormal("../data/ucd_tets.vtk")

    StartSection("testxform")
    TestTransform("../data/curv_cube.vtk")
    TestTransform("../data/curv_x.vtk")
    TestTransform("../data/curv_xy.vtk")
    TestTransform("../data/curv_xz.vtk")
    TestTransform("../data/curv_y.vtk")
    TestTransform("../data/curv_yz.vtk")
    TestTransform("../data/curv_z.vtk")
    TestTransform("../data/poly_1d_in_3d.vtk")
    TestTransform("../data/poly_2d_in_3d.vtk")
    TestTransform("../data/poly_sphere.vtk")
    TestTransform("../data/poly_verts_and_lines.vtk")
    TestTransform("../data/poly_verts.vtk")
    TestTransform("../data/rect_cube.vtk")
    TestTransform("../data/rect_x.vtk")
    TestTransform("../data/rect_xy.vtk")
    TestTransform("../data/rect_xz.vtk")
    TestTransform("../data/rect_y.vtk")
    TestTransform("../data/rect_yz.vtk")
    TestTransform("../data/rect_z.vtk")
    TestTransform("../data/ucd_1d_in_3d.vtk")
    TestTransform("../data/ucd_2d_xy.vtk")
    TestTransform("../data/ucd_cube.vtk")
    TestTransform("../data/ucd_sphere.vtk")
    TestTransform("../data/ucd_tets.vtk")

    StartSection("testdistancefield")
    TestDistanceField("../data/four_points_1d.vtk", [20], [-2], [7])
    TestDistanceField("../data/four_points_2d.vtk", [50,50], [-2,-2], [7,7])
    TestDistanceField("../data/four_points_3d.vtk", [20,20,20], [-2,-2,-2], [7,7,7])

    StartSection("testserialize")
    TestSerialization("../data/curv_cube.vtk")
    TestSerialization("../data/curv_x.vtk")
    TestSerialization("../data/curv_xy.vtk")
    TestSerialization("../data/curv_xz.vtk")        
    TestSerialization("../data/curv_y.vtk")
    TestSerialization("../data/curv_yz.vtk")    
    TestSerialization("../data/curv_z.vtk")
    TestSerialization("../data/poly_1d_in_3d.vtk")
    TestSerialization("../data/poly_2d_in_3d.vtk")
    TestSerialization("../data/poly_sphere.vtk")
    TestSerialization("../data/poly_verts.vtk")
    TestSerialization("../data/poly_verts_and_lines.vtk")
    TestSerialization("../data/rect_cube.vtk")
    TestSerialization("../data/rect_x.vtk")
    TestSerialization("../data/rect_xy.vtk")
    TestSerialization("../data/rect_xz.vtk")
    TestSerialization("../data/rect_y.vtk")
    TestSerialization("../data/rect_yz.vtk")
    TestSerialization("../data/rect_z.vtk")
    TestSerialization("../data/ucd_1d_in_3d.vtk")
    TestSerialization("../data/ucd_2d_xy.vtk")
    TestSerialization("../data/ucd_cube.vtk")
    TestSerialization("../data/ucd_sphere.vtk")
    TestSerialization("../data/ucd_tets.vtk")


    errors = PrintResults()

    logfile.close()
    difffile.close()
    tkdifffile.close()
    rebasefile.close()

    sys.exit(errors)

