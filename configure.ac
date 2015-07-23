dnl Process this file with autoconf to produce configure.

## ---------------------------------------------------------------------------
## Programmer: Jeremy Meredith
## Date:       December  1, 2010
##
## Modifications:
##
## ---------------------------------------------------------------------------

AC_INIT()
AC_CONFIG_HEADER(config/eavlConfig.h)

TOPDIR=`pwd`
AC_SUBST(TOPDIR)


## ---------------------------------------------------------------------------
## Load the OS configuration file if present.
## ---------------------------------------------------------------------------
UNAME=`uname 2>/dev/null`
SHORT_UNAME=`uname -s | tr A-Z a-z | tr -d 0-9`
site_conf=config-site/$UNAME.conf
AC_MSG_CHECKING([for OS configuration file])
if test -r $site_conf; then
    AC_MSG_RESULT($site_conf)
    . $site_conf
else
    AC_MSG_RESULT(none found)
fi


## ---------------------------------------------------------------------------
## Load the site configuration file if present.
## ---------------------------------------------------------------------------
AC_MSG_CHECKING(for site configuration file)

hostname=`hostname -f 2>/dev/null`
site_conf=config-site/$hostname.conf
if test -r $site_conf; then
   AC_MSG_RESULT($site_conf)
   . $site_conf
else
   ## Check for hostname without the -f
   hostnamenodashf=`hostname 2>/dev/null`
   site_conf=config-site/$hostnamenodashf.conf
   if test -r $site_conf; then
      AC_MSG_RESULT($site_conf)
      . $site_conf
   else
      ## Also check for hostnames where the first field is a node name.
      hostminusnodename=`echo $hostname | cut -d. -f2-`
      site_conf=config-site/$hostminusnodename.conf
      if test -r $site_conf; then
         AC_MSG_RESULT($site_conf)
         . $site_conf
      else
         ## Or a hostname where the first chunk has an ignore-able number.
         hostminusnodenumber=`echo $hostname | sed 's/^\([[_a-zA-Z\-]]\+\)[[0-9]]\+\./\1./'`
         site_conf=config-site/$hostminusnodenumber.conf
         if test -r $site_conf; then
            AC_MSG_RESULT($site_conf)
            . $site_conf
         else
            AC_MSG_RESULT(none found)
         fi
      fi
   fi
fi

## ---------------------------------------------------------------------------
## Load any manually specified configuration file.
## ---------------------------------------------------------------------------
AC_MSG_CHECKING(for manually specified configuration file)
AC_ARG_WITH(config,[
config-site Options:
  --with-config=path      specify location of config-site file manually],
        CONFIGSITE="$withval",
        CONFIGSITE="")
if test -z "$CONFIGSITE"; then
   AC_MSG_RESULT(none specified)
elif test -f "$CONFIGSITE"; then
   . $CONFIGSITE
   AC_MSG_RESULT($CONFIGSITE)
else
   AC_MSG_RESULT(not found)
fi


## ---------------------------------------------------------------------------
## Macro needed for runtime linker paths.
## ---------------------------------------------------------------------------
AC_DEFUN([rpathize],
         [if test "$UNAME" = "Darwin"; then
             $1="$$1"
          else
             $1="$$1 -Wl,-rpath=$2"
          fi])


## ---------------------------------------------------------------------------
## Check for programs.
## ---------------------------------------------------------------------------
AC_PROG_CXX(g++ c++ icc CC)
AC_PROG_CXXCPP
AC_PROG_CC(gcc icc cc)
AC_PROG_CPP
AC_PROG_MAKE_SET

AC_LANG(C++)


## ---------------------------------------------------------------------------
## Static or dynamic build.
## ---------------------------------------------------------------------------
AC_MSG_CHECKING(for static or dynamic library build)
AC_ARG_ENABLE(dynamic,[
Library Build Options:
  --enable-dynamic        enable dynamic library build, not static
  --disable-dynamic       enable static library build, not dynamic (DEFAULT)],
        DYNAMIC="$enableval",
        DYNAMIC="no")
if test "$DYNAMIC" = "no"; then
   AC_MSG_RESULT(static)
   LIBRARY_OR_ARCHIVE="$""(ARCHIVE)"
   LIB_NAME="libeavl.a"
elif test "$DYNAMIC" = "yes"; then
   AC_MSG_RESULT(dynamic)
   LIBRARY_OR_ARCHIVE="$""(LIBRARY)"
   LIB_NAME="libeavl.so"
else
   AC_MSG_RESULT(unknown value; assuming static)
   LIBRARY_OR_ARCHIVE="$""(ARCHIVE)"
   LIB_NAME="libeavl.a"
   #DYNAMIC="no"
fi
AC_SUBST(LIBRARY_OR_ARCHIVE)
AC_SUBST(LIB_NAME)


## ---------------------------------------------------------------------------
## Determine a dependency mode.
## ---------------------------------------------------------------------------
AC_MSG_CHECKING(for dependency mode)
if test "`${MAKE-make} --version -f /dev/null 2>/dev/null|sed -n 1p|cut -c1-8`" = "GNU Make"; then
   if test -n "$GCC"; then
      # Darwin's universal build doesn't like the make-depend-gnu
      if test "$UNAME" = "Darwin"; then
         DEPMODE=cpp
         MG=""
      else
         DEPMODE=gcc3
      fi
   elif test -n "$MAKEDEPEND"; then
      DEPMODE=cpp
   else
      DEPMODE=none
   fi
else
   if test -n "$MAKEDEPEND"; then
      DEPMODE=cpp
   else
      DEPMODE=none
   fi
fi
AC_MSG_RESULT($DEPMODE)
AC_SUBST(DEPMODE)


## ---------------------------------------------------------------------------
## MPI check.
## ---------------------------------------------------------------------------

# first, find a compiler to get flags from
MPICC=""
AC_ARG_WITH(mpi,[
MPI Options:
  --with-mpi              search for MPI
  --with-mpi=mpicc        specify MPI compiler to extract flags from
  --without-mpi           (DEFAULT) disable MPI],
        MPI="$withval",
        if test -z "$MPI"; then MPI="no"; fi)

detect_mpi_flags="no"
if test "$MPI" != "no"; then
   if test -n "$MPI_CPPFLAGS" -o -n "$MPI_LDFLAGS" -o -n "$MPI_LIBS"; then
      MPI="manually configured"
   else
      detect_mpi_flags="yes"
      if test "$MPI" = "yes"; then
         # search for mpi compiler, too
         MPI=""
         for mpicompiler in mpic++ mpicxx; do
            if test -z "$MPI"; then
               AC_PATH_PROG([MPI], [$mpicompiler], [], [])
            fi
         done
         if test -z "$MPI"; then
            AC_MSG_WARN(could not find MPI C++ compiler; disabling MPI)
            detect_mpi_flags="no"
            MPI="no"
         fi
      fi
   fi
fi

AC_MSG_CHECKING(for MPI)
AC_MSG_RESULT($MPI)

if test "$detect_mpi_flags" = "yes"; then
   # extract MPI compile info from compiler
   if test "$MPI" != "no"; then
      # option 1: -compile-info
      MPISHOW_C=`$MPI -compile-info 2>&AS_MESSAGE_LOG_FD`
      # option 2: -show
      if test "$?" -ne "0"; then
         MPISHOW_C=`$MPI -show 2>&AS_MESSAGE_LOG_FD`
      fi
      if test "$?" -ne "0"; then
         MPISHOW_C=`$MPI --cray-print-opts=cflags 2>&AS_MESSAGE_LOG_FD`
         if test "$?" -ne "0"; then
            MPI="no"
            AC_MSG_WARN(could not extract compile flags from MPI compiler; disabling MPI)
         fi
      fi
      
   fi

   # extract MPI link info from compiler
   if test "$MPI" != "no"; then
      # option 1: -link-info
      MPISHOW_L=`$MPI -link-info 2>&AS_MESSAGE_LOG_FD`
      # option 2: -show
      if test "$?" -ne "0"; then
         MPISHOW_L=`$MPI -show 2>&AS_MESSAGE_LOG_FD`
      fi
      if test "$?" -ne "0"; then
         MPISHOW_L=`$MPI --cray-print-opts=libs 2>&AS_MESSAGE_LOG_FD`
         if test "$?" -ne "0"; then
            MPI="no"
            AC_MSG_WARN(could not extract link flags from MPI compiler; disabling MPI)
         fi
      fi
   fi

   # turn raw MPI info into actual flags
   if test "$MPI" != "no"; then
      MPI_CPPFLAGS=`echo $MPISHOW_C | perl -e '$_=<>;print join(" ",grep(/^-I/,split(/\s+/)));'`
      #MPI_LDFLAGS=`echo $MPISHOW_L | perl -e '$_=<>;print join(" ",grep(!/^-I|^-l|^\w/,split(/\s+/)));'`
      MPI_LDFLAGS=`echo $MPISHOW_L | perl -e '$_=<>;print join(" ",grep(/^-L/,split(/\s+/)));'`
      MPI_LIBS=`echo $MPISHOW_L | perl -e '$_=<>;print join(" ",grep(/^-l/,split(/\s+/)));'`
   fi
fi

# now actually try MPI...
if test "$MPI" != "no"; then
   old_cppflags=$CPPFLAGS
   old_ldflags=$LDFLAGS
   old_libs=$LIBS
   CPPFLAGS="$CPPFLAGS $MPI_CPPFLAGS";
   LDFLAGS="$LDFLAGS $MPI_LDFLAGS";
   LIBS="$LIBS $MPI_LIBS"

   AC_MSG_CHECKING(if MPI works)
   AC_TRY_LINK([#include <mpi.h>],
               [int a; MPI_Bcast(&a,1,MPI_INT,0,MPI_COMM_WORLD);],
               AC_MSG_RESULT(yes),
               AC_MSG_RESULT(no);MPI="no";AC_MSG_WARN(could not get MPI to compile/link; disabling MPI))

   CPPFLAGS=$old_cppflags
   LDFLAGS=$old_ldflags
   LIBS=$old_libs

   AC_MSG_CHECKING(for MPI CPPFLAGS)
   AC_MSG_RESULT($MPI_CPPFLAGS)
   AC_MSG_CHECKING(for MPI LDFLAGS)
   AC_MSG_RESULT($MPI_LDFLAGS)
   AC_MSG_CHECKING(for MPI LIBS)
   AC_MSG_RESULT($MPI_LIBS)
fi

# set preprocessor definition
if test "$MPI" != "no"; then
   AC_DEFINE(HAVE_MPI,[1])
fi

AC_SUBST(MPI_CPPFLAGS)
AC_SUBST(MPI_LDFLAGS)
AC_SUBST(MPI_LIBS)


## ---------------------------------------------------------------------------
## OpenMP check.
## ---------------------------------------------------------------------------


OPENMP=""
AC_ARG_WITH(openmp,[
OpenMP Options:
  --with-openmp           enable OpenMP
  --with-openmp=flags     specify flags needed to enable OpenMP
  --without-openmp        (DEFAULT) disable OpenMP],
        OPENMP="$withval",
        if test -z "$OPENMP"; then OPENMP="no"; fi)

if test "$OPENMP" = "yes"; then
   OPENMP=""
   for ompflags in "-fopenmp" "-openmp"; do
      old_cxxflags=$CXXFLAGS
      # We usually get a warning if there's an unrecognized
      # pragma; with icc and gcc, forcing warnings to errors
      # actually ensures we're compiling with OpenMP.
      CXXFLAGS="$CXXFLAGS -Werror $ompflags";
      if test -z "$OPENMP"; then
         AC_MSG_CHECKING(if OpenMP support is enabled by $ompflags)
         AC_TRY_LINK([#include <omp.h>],
                     [int *x=new int[1024], *y=new int[1024];
                      #pragma omp parallel for
                      for (int i=0; i<1024; ++i) {x[i]=y[i]*2;}
                     ],
                     AC_MSG_RESULT(yes);OPENMP="$ompflags",
                     AC_MSG_RESULT(no))
      fi
      CXXFLAGS=$old_cxxflags
   done
   if test -z "$OPENMP"; then
      AC_MSG_WARN(attempts to enable OpenMP failed; disabling);
      OPENMP="no";
   fi
elif test "$OPENMP" != "no"; then
   old_cxxflags=$CXXFLAGS
   # Again, add -Werror to make check successful (see above).
   CXXFLAGS="$CXXFLAGS -Werror $OPENMP"
   AC_MSG_CHECKING(if OpenMP support is enabled by $OPENMP)
   AC_TRY_LINK([#include <omp.h>],
               [int *x=new int[1024], *y=new int[1024];
                #pragma omp parallel for
                for (int i=0; i<1024; ++i) {x[i]=y[i]*2;}
               ],
               AC_MSG_RESULT(yes),
               AC_MSG_RESULT(no);OPENMP="no";AC_MSG_WARN(given flags did not enable OpenMP; disabling))
   CXXFLAGS=$old_cxxflags
fi

# We're going to use CXXFLAGS for the OpenMP flags
if test "$OPENMP" != "no"; then
   AC_DEFINE(HAVE_OPENMP,[1])
   CXXFLAGS="$CXXFLAGS $OPENMP"
fi

AC_MSG_CHECKING(for OpenMP)
AC_MSG_RESULT($OPENMP)


## ---------------------------------------------------------------------------
## Boost check.
## ---------------------------------------------------------------------------
AC_MSG_CHECKING(for Boost)

## \TODO: We do not YET use boost in EAVL without MPI.
## Remove the "--without-boost" help text option entirely when we do need it.
AC_ARG_WITH(boost,[
Boost Options:
  --with-boost             assume compiler knows about Boost
  --with-boost=path        specify location of Boost installation
  --without-boost          (DEFAULT) disable Boost support; note that at
                           the moment, all Boost usage has been removed
                           from EAVL to minimize dependencies]],
        BOOST="$withval",
        if test -z "$BOOST"; then BOOST="no"; fi)
AC_MSG_RESULT($BOOST)

## \TODO: We have removed all boost from EAVL for now.
## Uncomment the next 3 lines if we do need it.
#if test "$BOOST" = "no"; then
#   AC_MSG_ERROR(Boost is required)
#fi
# Also, this entire section of code can have
# the "if boost != no" check removed if we
# require boost again, because we already
# know boost != no if we enabled the above check:
if test "$BOOST" != "no"; then
   old_cppflags=$CPPFLAGS
   old_ldflags=$LDFLAGS
   old_libs=$LIBS
   
   # cppflags and basic headers
   BOOST_CPPFLAGS=""
   if test "$BOOST" != "yes"; then
      BOOST_CPPFLAGS="-I${BOOST}/include"
   fi
   
   AC_MSG_CHECKING(if Boost basic headers work)
   CPPFLAGS="$old_cppflags $BOOST_CPPFLAGS"
   AC_TRY_COMPILE([#include <boost/logic/tribool.hpp>],
                  [boost::tribool tb(false);],
                  AC_MSG_RESULT(yes),
                  AC_MSG_RESULT(no))
   ## \TODO: We have removed all boost from EAVL for now.
   ## Remove the preceeding line and replace it with this next
   ## line when we do need it (i.e. add the fatal error back in).
   #              AC_MSG_RESULT(no);AC_MSG_ERROR(Boost is required))
   
   # ldflags and some libraries
   BOOST_LDFLAGS=""
   if test "$BOOST" != "yes"; then
      BOOST_LDFLAGS="-L${BOOST}/lib"
      rpathize(BOOST_LDFLAGS, ${BOOST}/lib)
   fi
   LDFLAGS="$old_ldflags $BOOST_LDFLAGS"
   
   BOOST_LIBS=""
   
   AC_MSG_CHECKING(if Boost regex library works)
   tmplib="-lboost_regex"
   LIBS="$old_libs $tmplib"
   AC_TRY_LINK([#include <boost/regex.hpp>],
               [static const boost::regex re("\\\\d{4}");
                return boost::regex_match("foo1234",re);],
               AC_MSG_RESULT(yes);BOOST_LIBS="$BOOST_LIBS $tmplib",
               AC_MSG_RESULT(no))
   ## \TODO: We have removed all boost from EAVL for now.
   ## Remove the preceeding line and replace it with this next
   ## line when we do need it (i.e. add the fatal error back in).
   #            AC_MSG_RESULT(no);AC_MSG_ERROR(Boost is required))
   
   # MPI requires serialization; if we're not
   # using MPI, just check serialization explicitly
   if test "$MPI" = "no"; then
      AC_MSG_CHECKING(if Boost serialization library works)
      tmplib="-lboost_serialization"
      LIBS="$old_libs $tmplib"
      AC_TRY_LINK([#include <boost/archive/text_oarchive.hpp>
                   #include <fstream>],
                  [std::ofstream out("tmp.txt");
                   boost::archive::text_oarchive ar(out);
                   int i=105; ar & i;],
                  AC_MSG_RESULT(yes);BOOST_LIBS="$BOOST_LIBS $tmplib",
                  AC_MSG_RESULT(no))
       ## \TODO: We have removed all boost from EAVL for now.
       ## Remove the preceeding line and replace it with this next
       ## line when we do need it (i.e. add the fatal error back in).
       #          AC_MSG_RESULT(no);AC_MSG_ERROR(Boost is required))
   else
      AC_MSG_CHECKING(if Boost MPI and serialization libraries work)
      tmplib="-lboost_mpi -lboost_serialization"
      CPPFLAGS="$CPPFLAGS $MPI_CPPFLAGS";
      LDFLAGS="$LDFLAGS $MPI_LDFLAGS";
      LIBS="$old_libs $tmplib $MPI_LIBS"
      AC_TRY_LINK([#include <boost/mpi.hpp>],
                  [boost::mpi::communicator world;],
                  AC_MSG_RESULT(yes);BOOST_LIBS="$BOOST_LIBS $tmplib",
                  AC_MSG_RESULT(no))
       ## \TODO: We have removed all boost from EAVL for now.
       ## Remove the preceeding line and replace it with this next
       ## line when we do need it (i.e. add the fatal error back in).
       #          AC_MSG_RESULT(no);AC_MSG_ERROR(Boost is required))
   fi
   
   CPPFLAGS=$old_cppflags
   LDFLAGS=$old_ldflags
   LIBS=$old_libs
fi
   
AC_SUBST(BOOST)
AC_SUBST(BOOST_CPPFLAGS)
AC_SUBST(BOOST_LDFLAGS)
AC_SUBST(BOOST_LIBS)

## ---------------------------------------------------------------------------
## TR1 check.
## ---------------------------------------------------------------------------

AC_MSG_CHECKING(for TR1 headers from the compiler)
compiler_has_tr1="no"
AC_TRY_COMPILE([#include <tr1/unordered_map>],
               [std::tr1::unordered_map<int,double> hashtable;],
               compiler_has_tr1="yes")
AC_MSG_RESULT($compiler_has_tr1)

# fall back to the boost version
if test "$compiler_has_tr1" = "no"; then
   old_cppflags=$CPPFLAGS
   AC_MSG_CHECKING(if TR1 headers in Boost can be used instead)
   BOOST_CPPFLAGS="-I${BOOST}/include/boost/tr1 $BOOST_CPPFLAGS"
   CPPFLAGS="$old_cppflags $BOOST_CPPFLAGS"
   AC_TRY_COMPILE([#include <tr1/unordered_map>],
                  [std::tr1::unordered_map<int,double> hashtable;],
                  AC_MSG_RESULT(yes),
                  AC_MSG_RESULT(no))
    ## \TODO: We have removed all TR1 usage from EAVL for now.
    ## Remove the preceeding line and replace it with this next
    ## line when we do need it (i.e. add the fatal error back in).
    #             AC_MSG_RESULT(no);AC_MSG_ERROR(TR1 headers are required))
   CPPFLAGS=$old_cppflags
fi


## ---------------------------------------------------------------------------
## CUDA check.
## ---------------------------------------------------------------------------
NVCC=""
AC_ARG_WITH(cuda,[
CUDA Options:
  --with-cuda             (DEFAULT) search for CUDA
  --with-cuda=path        specify location of CUDA installation
  --without-cuda          disable CUDA],
        CUDA="$withval",
        if test -z "$CUDA"; then CUDA="yes"; fi)
if test "$CUDA" = "yes"; then
   # gotta search
   AC_PATH_PROG([NVCC], [nvcc], [],
                [PATH="$PATH:/usr/local/cuda/bin:/opt/cuda/bin:/usr/local/cuda40/cuda/bin:/usr/local/cuda41/cuda/bin:/usr/local/cuda42/cuda/bin::/usr/local/cuda50/cuda/bin::/usr/local/cuda60/cuda/bin::/usr/local/cuda65/cuda/bin::/usr/local/cuda70/cuda/bin"])
   if test -z "$NVCC"; then
      AC_MSG_WARN(disabling CUDA)
      CUDA="no"
   else
     CUDABIN=`AS_DIRNAME(["$NVCC"])`
     CUDA=`AS_DIRNAME(["$CUDABIN"])`
   fi
elif test "$CUDA" != "no"; then
   # path was given; we'll check in a sec if it's right
   CUDABIN="$CUDA/bin"
   NVCC="$CUDABIN/nvcc"
fi

# does nvcc work?
if test "$CUDA" != "no"; then
   AC_MSG_CHECKING(if nvcc works)
   echo "__global__ void testfunc() { }" > conftest.cu
   if $NVCC -c conftest.cu 2>&AS_MESSAGE_LOG_FD ; then
      AC_MSG_RESULT(yes)
      AC_DEFINE(HAVE_CUDA,[1])
   else
      AC_MSG_RESULT(no)
      AC_MSG_WARN(disabling CUDA)
      CUDA="no"
      NVCC=""
   fi
fi

# if nvcc works, set up general include and linker paths
if test "$CUDA" != "no"; then
   CUDA_CPPFLAGS="-I""$CUDA""/include"
   CUDA_LDFLAGS="-L""$CUDA""/lib"
   rpathize(CUDA_LDFLAGS, $CUDA/lib)
   if test -z "$CUDA_LIBS"; then
      CUDA_LIBS="-lcuda -lcudart -lcurand"
   fi
   old_cppflags=$CPPFLAGS
   old_ldflags=$LDFLAGS
   old_libs=$LIBS
   CPPFLAGS="$CPPFLAGS $CUDA_CPPFLAGS";
   LDFLAGS="$LDFLAGS $CUDA_LDFLAGS";
   LIBS="$LIBS $CUDA_LIBS"

   AC_MSG_CHECKING(if CUDA include and linker paths work with lib/)
   AC_TRY_LINK([#include <cuda.h>
                #include <cuda_runtime_api.h>],
               [ float *vals; cudaMalloc((void**)&vals, 4096); ],
               AC_MSG_RESULT(yes);tryagain="no",
               AC_MSG_RESULT(no);tryagain="yes")

   if test "$tryagain" = "yes"; then
      CUDA_LDFLAGS="-L""$CUDA""/lib64"
      rpathize(CUDA_LDFLAGS, $CUDA/lib64)
      LDFLAGS="$old_ldflags $CUDA_LDFLAGS"

      AC_MSG_CHECKING(if CUDA include and linker paths work with lib64/)
      AC_TRY_LINK([#include <cuda.h>
                   #include <cuda_runtime_api.h>],
                  [ float *vals; cudaMalloc((void**)&vals, 4096); ],
                  AC_MSG_RESULT(yes),
                  AC_MSG_RESULT(no);CUDA="no";AC_MSG_WARN(CUDA would not link with either lib/ or lib64/; disabling CUDA.))
   fi

   CPPFLAGS=$old_cppflags
   LDFLAGS=$old_ldflags
   LIBS=$old_libs
fi

# set up flags for nvcc
if test "$CUDA" != "no"; then

   # We don't want to interpret the CXXFLAGS value right now, so we
   # need to have it pass the CXXFLAGS variable to be interpolated at
   # the last minute.
   # However, nvcc needs -Xcompiler flag prefacing to all of the flags.
   # Something in depcomp isn't quoting the CXXFLAGS properly (and it
   # doesn't look easily fixable properly), or else just double-quoting
   # CXXFLAGS here would work.  However, -Xcompiler will accept comma-
   # separated flags, so we just have it translate spaces to commas
   # at the last minute.
   NVCXXFLAGS="$NVCXXFLAGS -Xcompiler \`echo \"$""(CXXFLAGS)\" | tr ' ' ','\`"

   # find the latest compute model
   AC_MSG_CHECKING(for latest compute model supported)
   echo "__global__ void testfunc() { }" > conftest.cu
   if $NVCC -gencode=arch=compute_35,code=sm_35 -c conftest.cu 2>&AS_MESSAGE_LOG_FD ; then
      AC_MSG_RESULT(3.5)
      NVCXXFLAGS="$NVCXXFLAGS -gencode=arch=compute_20,code=sm_20"
      NVCXXFLAGS="$NVCXXFLAGS -gencode=arch=compute_30,code=sm_30"
      NVCXXFLAGS="$NVCXXFLAGS -gencode=arch=compute_35,code=sm_35"
      NVCXXFLAGS="$NVCXXFLAGS -gencode=arch=compute_35,code=compute_35"
   elif $NVCC -gencode=arch=compute_30,code=sm_30 -c conftest.cu 2>&AS_MESSAGE_LOG_FD ; then
      AC_MSG_RESULT(3.0)
      NVCXXFLAGS="$NVCXXFLAGS -gencode=arch=compute_20,code=sm_20"
      NVCXXFLAGS="$NVCXXFLAGS -gencode=arch=compute_30,code=sm_30"
      NVCXXFLAGS="$NVCXXFLAGS -gencode=arch=compute_30,code=compute_30"
   elif $NVCC -gencode=arch=compute_20,code=sm_20 -c conftest.cu 2>&AS_MESSAGE_LOG_FD ; then
      AC_MSG_RESULT(2.0)
      NVCXXFLAGS="$NVCXXFLAGS -gencode=arch=compute_20,code=sm_20"
      NVCXXFLAGS="$NVCXXFLAGS -gencode=arch=compute_20,code=compute_20"
   else
      AC_MSG_RESULT(older than 2.0)
      AC_MSG_ERROR([EAVL requires compute capability 2.0 or later.  Please upgrade your CUDA version.])
   fi
fi

AC_MSG_CHECKING(for CUDA)
AC_MSG_RESULT($CUDA)

AC_SUBST(NVCC)
AC_SUBST(NVCXXFLAGS)
AC_SUBST(CUDA)
AC_SUBST(CUDA_CPPFLAGS)
AC_SUBST(CUDA_LDFLAGS)
AC_SUBST(CUDA_LIBS)


## ---------------------------------------------------------------------------
## VTK check.
## ---------------------------------------------------------------------------

# don't use everything -- we only need enough to do data set conversion.
# also, we'll test each individually, so it's safe to have both 6.0 and 6.1 here
VTK_ALL_LIBS="
    vtkIOLegacy-6.0
    vtkIOCore-6.0
    vtkCommonDataModel-6.0
    vtkCommonCore-6.0
    vtkIOLegacy-6.1
    vtkIOCore-6.1
    vtkCommonDataModel-6.1
    vtkCommonCore-6.1
    "

VTK_CPPFLAGS=""
VTK_LDFLAGS=""
VTK_LIBS=""
AC_ARG_WITH(vtk,[
VTK Options:
  --with-vtk              try VTK with no paths
  --with-vtk=path         specify location of VTK installation
  --without-vtk           (DEFAULT) disable VTK],
        VTK="$withval",
        if test -z "$VTK"; then VTK="no"; fi)
if test "$VTK" != "yes" -a "$VTK" != "no"; then
   # specified path

   # figure out if they're on 6.0 or 6.1
   AC_MSG_CHECKING(for attempted VTK version via include path)
   if test -d "$VTK""/include/vtk-6.0"; then 
      VTK_CPPFLAGS="-I""$VTK""/include/vtk-6.0 -I""$VTK""/include"
      AC_MSG_RESULT(6.0)
   elif test -d "$VTK""/include/vtk-6.1"; then 
      VTK_CPPFLAGS="-I""$VTK""/include/vtk-6.1 -I""$VTK""/include"
      AC_MSG_RESULT(6.1)
   else
      VTK_CPPFLAGS="-I""$VTK""/include"
      AC_MSG_RESULT(Unknown. This will probably fail to get a working VTK.)
   fi

   VTK_LDFLAGS="-L""$VTK""/lib"
   rpathize(VTK_LDFLAGS, $VTK/lib)
   VTK_LIBS=""
   for lib in $VTK_ALL_LIBS; do
      if test -f "$VTK/lib/lib$lib.so" -o \
              -f "$VTK/lib/lib$lib.a" -o \
              -f "$VTK/lib/lib$lib.dylib"; then
         VTK_LIBS="$VTK_LIBS -l$lib"
      fi
   done
fi

# make sure it works with those flags
if test "$VTK" != "no"; then
   old_cppflags=$CPPFLAGS
   old_ldflags=$LDFLAGS
   old_libs=$LIBS
   CPPFLAGS="$CPPFLAGS $VTK_CPPFLAGS";
   LDFLAGS="$LDFLAGS $VTK_LDFLAGS";
   LIBS="$LIBS $VTK_LIBS"

   AC_MSG_CHECKING(if VTK works)
   AC_TRY_LINK([#include <vtkPolyData.h>],
               [ vtkPolyData::New() ],
               AC_MSG_RESULT(yes),
               AC_MSG_RESULT(no);VTK="no";AC_MSG_WARN(could not get VTK to compile/link; disabling VTK))

   CPPFLAGS=$old_cppflags
   LDFLAGS=$old_ldflags
   LIBS=$old_libs
fi

# set preprocessor definition
if test "$VTK" != "no"; then
   AC_DEFINE(HAVE_VTK,[1])
fi


AC_MSG_CHECKING(for VTK)
AC_MSG_RESULT($VTK)
AC_SUBST(VTK)
AC_SUBST(VTK_CPPFLAGS)
AC_SUBST(VTK_LDFLAGS)
AC_SUBST(VTK_LIBS)


## ---------------------------------------------------------------------------
## Static or dynamic build.
## ---------------------------------------------------------------------------
AC_MSG_CHECKING(for I/O library support)
AC_ARG_ENABLE(io,[
Library Build Options:
  --enable-io             enable found I/O libraries (DEFAULT)
  --disable-io            disable all I/O library support, typically
                          useful for embedding in situ or within a tool
                          which has its own file I/O support],
        ALLOWIO="$enableval",
        ALLOWIO="yes")

if test "$ALLOWIO" = "no"; then
   AC_MSG_RESULT(no)
elif test "$ALLOWIO" = "yes"; then
   AC_MSG_RESULT(yes)
else
   AC_MSG_RESULT(unknown value; assuming yes)
   ALLOWIO="no"
fi

if test "$ALLOWIO" = "no"; then
  SZIP="no"
  ZLIB="no"
  HDF5="no"
  NETCDF="no"
  SILO="no"
  ADIOS="no"
fi


## ---------------------------------------------------------------------------
## SZIP check
## ---------------------------------------------------------------------------
SZIP_CPPFLAGS=""
SZIP_LDFLAGS=""
SZIP_LIBS=""
AC_ARG_WITH(szip,[
SZIP Options:
  --with-szip             (DEFAULT) try SZIP with no paths
  --with-szip=path        specify location of SZIP installation
  --without-szip          disable SZIP],
        SZIP="$withval",
        if test -z "$SZIP"; then SZIP="yes"; fi)
if test "$SZIP" != "yes" -a "$SZIP" != "no"; then
   # specified path
   SZIP_CPPFLAGS="-I""$SZIP""/include"
   SZIP_LDFLAGS="-L""$SZIP""/lib"
   rpathize(SZIP_LDFLAGS, $SZIP/lib)
fi

if test "$SZIP" != "no"; then
   SZIP_LIBS="-lsz -lz"
fi

# make sure it works with those flags
if test "$SZIP" != "no"; then
   old_cppflags=$CPPFLAGS
   old_ldflags=$LDFLAGS
   old_libs=$LIBS
   CPPFLAGS="$CPPFLAGS $SZIP_CPPFLAGS";
   LDFLAGS="$LDFLAGS $SZIP_LDFLAGS";
   LIBS="$LIBS $SZIP_LIBS"

   AC_MSG_CHECKING(if SZIP works)
   AC_TRY_LINK([#include <stdlib.h>
                extern "C" {
                #include <szlib.h>
                }],
               [int enabled = SZ_encoder_enabled(); ],
               AC_MSG_RESULT(yes),
               AC_MSG_RESULT(no);SZIP="no";AC_MSG_WARN(could not get SZIP to compile/link; disabling SZIP))

   CPPFLAGS=$old_cppflags
   LDFLAGS=$old_ldflags
   LIBS=$old_libs
fi

# set preprocessor definition
if test "$SZIP" != "no"; then
   AC_DEFINE(HAVE_SZIP,[1])
else
   SZIP_CPPFLAGS=""
   SZIP_LDFLAGS=""
   SZIP_LIBS=""
fi

AC_MSG_CHECKING(for SZIP)
AC_MSG_RESULT($SZIP)
AC_SUBST(SZIP)
AC_SUBST(SZIP_CPPFLAGS)
AC_SUBST(SZIP_LDFLAGS)
AC_SUBST(SZIP_LIBS)


## ---------------------------------------------------------------------------
## ZLIB check
## ---------------------------------------------------------------------------
ZLIB_CPPFLAGS=""
ZLIB_LDFLAGS=""
ZLIB_LIBS=""
AC_ARG_WITH(zlib,[
ZLIB Options:
  --with-zlib             (DEFAULT) try ZLIB with no paths
  --with-zlib=path        specify location of ZLIB installation
  --without-zlib          disable ZLIB],
        ZLIB="$withval",
        if test -z "$ZLIB"; then ZLIB="yes"; fi)
if test "$ZLIB" != "yes" -a "$ZLIB" != "no"; then
   # specified path
   ZLIB_CPPFLAGS="-I""$ZLIB""/include"
   ZLIB_LDFLAGS="-L""$ZLIB""/lib"
   rpathize(ZLIB_LDFLAGS, $ZLIB/lib)
fi

if test "$ZLIB" != "no"; then
   ZLIB_LIBS="-lz"
fi

# make sure it works with those flags
if test "$ZLIB" != "no"; then
   old_cppflags=$CPPFLAGS
   old_ldflags=$LDFLAGS
   old_libs=$LIBS
   CPPFLAGS="$CPPFLAGS $ZLIB_CPPFLAGS";
   LDFLAGS="$LDFLAGS $ZLIB_LDFLAGS";
   LIBS="$LIBS $ZLIB_LIBS"

   AC_MSG_CHECKING(if ZLIB works)
   AC_TRY_LINK([#include <zlib.h>],
               [void *fp = gzopen("filename", "r"); ],
               AC_MSG_RESULT(yes),
               AC_MSG_RESULT(no);ZLIB="no";AC_MSG_WARN(could not get ZLIB to compile/link; disabling ZLIB))

   CPPFLAGS=$old_cppflags
   LDFLAGS=$old_ldflags
   LIBS=$old_libs
fi

# set preprocessor definition
if test "$ZLIB" != "no"; then
   AC_DEFINE(HAVE_ZLIB,[1])
else
   ZLIB_CPPFLAGS=""
   ZLIB_LDFLAGS=""
   ZLIB_LIBS=""
fi

AC_MSG_CHECKING(for ZLIB)
AC_MSG_RESULT($ZLIB)
AC_SUBST(ZLIB)
AC_SUBST(ZLIB_CPPFLAGS)
AC_SUBST(ZLIB_LDFLAGS)
AC_SUBST(ZLIB_LIBS)


## ---------------------------------------------------------------------------
## HDF5 check
## ---------------------------------------------------------------------------
HDF5_CPPFLAGS=""
HDF5_LDFLAGS=""
HDF5_LIBS=""
AC_ARG_WITH(hdf5,[
HDF5 Options:
  --with-hdf5             (DEFAULT) try HDF5 with no paths
  --with-hdf5=path        specify location of HDF5 installation
  --without-hdf5          disable HDF5],
        HDF5="$withval",
        if test -z "$HDF5"; then HDF5="yes"; fi)
if test "$HDF5" != "yes" -a "$HDF5" != "no"; then
   # specified path
   HDF5_CPPFLAGS="-I""$HDF5""/include"
   HDF5_LDFLAGS="-L""$HDF5""/lib"
   rpathize(HDF5_LDFLAGS, $HDF5/lib)
   HDF5_LIBS="-lhdf5 -lhdf5_hl"
   # note: the HL libraries are mostly for NetCDF4
fi

# make sure it works with those flags
if test "$HDF5" != "no"; then
   old_cppflags=$CPPFLAGS
   old_ldflags=$LDFLAGS
   old_libs=$LIBS
   CPPFLAGS="$CPPFLAGS $HDF5_CPPFLAGS";
   LDFLAGS="$LDFLAGS $HDF5_LDFLAGS";
   LIBS="$LIBS $HDF5_LIBS"

   AC_MSG_CHECKING(if HDF5 works)
   AC_TRY_LINK([#include <hdf5.h>],
               [hid_t fileId = H5Fopen("/path/to/file.h5", H5F_ACC_RDONLY, H5P_DEFAULT); ],
               AC_MSG_RESULT(yes),
               AC_MSG_RESULT(no);HDF5="no";AC_MSG_WARN(could not get HDF5 to compile/link; disabling HDF5))

   CPPFLAGS=$old_cppflags
   LDFLAGS=$old_ldflags
   LIBS=$old_libs
fi

# set preprocessor definition
if test "$HDF5" != "no"; then
   AC_DEFINE(HAVE_HDF5,[1])
else
   HDF5_CPPFLAGS=""
   HDF5_LDFLAGS=""
   HDF5_LIBS=""
fi


AC_MSG_CHECKING(for HDF5)
AC_MSG_RESULT($HDF5)
AC_SUBST(HDF5)
AC_SUBST(HDF5_CPPFLAGS)
AC_SUBST(HDF5_LDFLAGS)
AC_SUBST(HDF5_LIBS)


## ---------------------------------------------------------------------------
## SILO check
## NOTE: Assuming HDF5 and SZIP dependency
## ---------------------------------------------------------------------------
SILO_CPPFLAGS=""
SILO_LDFLAGS=""
SILO_LIBS=""
AC_ARG_WITH(silo,[
SILO Options:
  --with-silo             (DEFAULT) try Silo with no paths
  --with-silo=path        specify location of Silo installation
  --without-silo          disable Silo],
        SILO="$withval",
        if test -z "$SILO"; then SILO="yes"; fi)
if test "$SILO" != "yes" -a "$SILO" != "no"; then
   # specified path
   SILO_CPPFLAGS="-I""$SILO""/include"
   SILO_LDFLAGS="-L""$SILO""/lib"
   rpathize(SILO_LDFLAGS, $SILO/lib)
   SILO_LIBS="-lsilo"
fi

# make sure it works with those flags
if test "$SILO" != "no"; then
   old_cppflags=$CPPFLAGS
   old_ldflags=$LDFLAGS
   old_libs=$LIBS
   CPPFLAGS="$CPPFLAGS $SILO_CPPFLAGS";
   LDFLAGS="$LDFLAGS $SILO_LDFLAGS $HDF5_LDFLAGS $SZIP_LDFLAGS";
   LIBS="$LIBS $SILO_LIBS $HDF5_LIBS $SZIP_LIBS"

   AC_MSG_CHECKING(if Silo works with -lsilo)
   AC_TRY_LINK([#include <silo.h>],
               [int err = DBErrno(); ],
               AC_MSG_RESULT(yes);tryagain="no",
               AC_MSG_RESULT(no);tryagain="yes")

   if test "$tryagain" = "yes"; then
      SILO_LIBS="-lsiloh5"
      LIBS="$old_libs $SILO_LIBS $HDF5_LIBS $SZIP_LIBS"
      AC_MSG_CHECKING(if Silo works with -lsiloh5)
      AC_TRY_LINK([#include <silo.h>],
                  [int err = DBErrno(); ],
                  AC_MSG_RESULT(yes),
                  AC_MSG_RESULT(no);SILO="no";AC_MSG_WARN(could not get Silo to compile/link using -lsiloh5 either; disabling Silo.))
   fi

   CPPFLAGS=$old_cppflags
   LDFLAGS=$old_ldflags
   LIBS=$old_libs
fi

# set preprocessor definition
if test "$SILO" != "no"; then
   AC_DEFINE(HAVE_SILO,[1])
else
   SILO_CPPFLAGS=""
   SILO_LDFLAGS=""
   SILO_LIBS=""
fi

AC_MSG_CHECKING(for Silo)
AC_MSG_RESULT($SILO)
AC_SUBST(SILO)
AC_SUBST(SILO_CPPFLAGS)
AC_SUBST(SILO_LDFLAGS)
AC_SUBST(SILO_LIBS)


## ---------------------------------------------------------------------------
## NetCDF check
## NOTE: Assuming HDF5 dependency
## ---------------------------------------------------------------------------
NETCDF_CPPFLAGS=""
NETCDF_LDFLAGS=""
NETCDF_LIBS=""
AC_ARG_WITH(netcdf,[
NetCDF Options:
  --with-netcdf           (DEFAULT) try NetCDF with no paths
  --with-netcdf=path      specify location of NetCDF installation
  --without-netcdf        disable NetCDF],
        NETCDF="$withval",
        if test -z "$NETCDF"; then NETCDF="yes"; fi)
if test "$NETCDF" != "yes" -a "$NETCDF" != "no"; then
   # specified path
   NETCDF_CPPFLAGS="-I""$NETCDF""/include"
   NETCDF_LDFLAGS="-L""$NETCDF""/lib"
   rpathize(NETCDF_LDFLAGS, $NETCDF/lib)
   NETCDF_LIBS="-lnetcdf_c++ -lnetcdf"
fi

# make sure it works with those flags
if test "$NETCDF" != "no"; then
   old_cppflags=$CPPFLAGS
   old_ldflags=$LDFLAGS
   old_libs=$LIBS
   CPPFLAGS="$CPPFLAGS $NETCDF_CPPFLAGS";
   LDFLAGS="$LDFLAGS $NETCDF_LDFLAGS $HDF5_LDFLAGS";
   LIBS="$LIBS $NETCDF_LIBS $HDF5_LIBS"

   AC_MSG_CHECKING(if NETCDF works)
   AC_TRY_LINK([#include <netcdfcpp.h>],
               [NcFile nf("/path/to/file.nc"); ],
               AC_MSG_RESULT(yes),
               AC_MSG_RESULT(no);NETCDF="no";AC_MSG_WARN(could not get NetCDF to compile/link; note it may require HDF5 to be enabled; disabling NetCDF))

   CPPFLAGS=$old_cppflags
   LDFLAGS=$old_ldflags
   LIBS=$old_libs
fi

# set preprocessor definition
if test "$NETCDF" != "no"; then
   AC_DEFINE(HAVE_NETCDF,[1])
else
   NETCDF_CPPFLAGS=""
   NETCDF_LDFLAGS=""
   NETCDF_LIBS=""
fi

AC_MSG_CHECKING(for NetCDF with C++ API)
AC_MSG_RESULT($NETCDF)
AC_SUBST(NETCDF)
AC_SUBST(NETCDF_CPPFLAGS)
AC_SUBST(NETCDF_LDFLAGS)
AC_SUBST(NETCDF_LIBS)


## ---------------------------------------------------------------------------
## ADIOS check
## ---------------------------------------------------------------------------
ADIOS_CPPFLAGS=""
ADIOS_LDFLAGS=""
ADIOS_LIBS=""
AC_ARG_WITH(adios,[
ADIOS Options:
  --with-adios             (DEFAULT) try ADIOS with no paths
  --with-adios=path        specify location of ADIOS installation
  --without-adios          disable ADIOS],
        ADIOS="$withval",
        if test -z "$ADIOS"; then ADIOS="yes"; fi)
if test "$ADIOS" != "yes" -a "$ADIOS" != "no"; then
   # specified path
   ADIOS_CPPFLAGS="-I""$ADIOS""/include $MPI_CPPFLAGS"
   ADIOS_LDFLAGS="-L""$ADIOS""/lib $MPI_LDFLAGS"
   rpathize(ADIOS_LDFLAGS, $ADIOS/lib)

   if test "$MPI" != "no"; then
     ADIOS_LIBS="-ladiosread $MPI_LIBS"
   else
     ADIOS_LIBS="-ladiosread_nompi"
   fi
fi

# make sure it works with those flags
if test "$ADIOS" != "no"; then
   old_cppflags=$CPPFLAGS
   old_ldflags=$LDFLAGS
   old_libs=$LIBS
   CPPFLAGS="$CPPFLAGS $ADIOS_CPPFLAGS -D_NOMPI";
   LDFLAGS="$LDFLAGS $ADIOS_LDFLAGS";
   LIBS="$LIBS $ADIOS_LIBS"

   AC_MSG_CHECKING(if ADIOS works)
   AC_TRY_LINK([#include <adios_read.h>],
               [ADIOS_FILE *f = adios_read_open_stream("/path/to/file.bp", ADIOS_READ_METHOD_BP, 0, ADIOS_LOCKMODE_ALL, 0); ],
               AC_MSG_RESULT(yes),
               AC_MSG_RESULT(no);ADIOS="no";AC_MSG_WARN(could not get ADIOS to compile/link; disabling ADIOS))

   CPPFLAGS=$old_cppflags
   LDFLAGS=$old_ldflags
   LIBS=$old_libs
fi

# set preprocessor definition
if test "$ADIOS" != "no"; then
   AC_DEFINE(HAVE_ADIOS,[1])
else
   ADIOS_CPPFLAGS=""
   ADIOS_LDFLAGS=""
   ADIOS_LIBS=""
fi

AC_MSG_CHECKING(for ADIOS)
AC_MSG_RESULT($ADIOS)
AC_SUBST(ADIOS)
AC_SUBST(ADIOS_CPPFLAGS)
AC_SUBST(ADIOS_LDFLAGS)
AC_SUBST(ADIOS_LIBS)



## ---------------------------------------------------------------------------
## Mesa check
## ---------------------------------------------------------------------------
MESA_CPPFLAGS=""
MESA_LDFLAGS=""
MESA_LIBS=""
AC_ARG_WITH(mesa,[
Mesa Options:
  --with-mesa             (DEFAULT) try Mesa with no paths
  --with-mesa=path        specify location of ADIOS installation
  --without-mesa          disable Mesa],
        MESA="$withval",
        if test -z "$MESA"; then MESA="yes"; fi)
if test "$MESA" != "yes" -a "$MESA" != "no"; then
   # specified path
   MESA_CPPFLAGS="-I""$MESA""/include"
   MESA_LDFLAGS="-L""$MESA""/lib $MPI_LDFLAGS"
   rpathize(MESA_LDFLAGS, $MESA/lib)
   MESA_LIBS="-lOSMesa"
fi

# make sure it works with those flags
if test "$MESA" != "no"; then
   old_cppflags=$CPPFLAGS
   old_ldflags=$LDFLAGS
   old_libs=$LIBS
   CPPFLAGS="$CPPFLAGS $MESA_CPPFLAGS";
   LDFLAGS="$LDFLAGS $MESA_LDFLAGS";
   LIBS="$LIBS $MESA_LIBS"

   AC_MSG_CHECKING(if MESA works)
   AC_TRY_LINK([#include <GL/osmesa.h>],
   	       [OSMesaContext ctx = OSMesaGetCurrentContext(); glEnable(GL_ALPHA_TEST); ],
               AC_MSG_RESULT(yes);tryMangledMesa="no",
               AC_MSG_RESULT(no);tryMangledMesa="yes")
   if test "$tryMangledMesa" = "yes"; then
      AC_MSG_CHECKING(if Mangled MESA works)
      CPPFLAGS="$CPPFLAGS -DUSE_MGL_NAMESPACE";
      AC_TRY_LINK([#include <GL/osmesa.h>],
   	       [OSMesaContext ctx = OSMesaGetCurrentContext(); glEnable(GL_ALPHA_TEST); ],
               AC_MSG_RESULT(yes),
               AC_MSG_RESULT(no);MESA="no";AC_MSG_WARN(could not get Mesa to compile/link; disabling Mesa))
   fi

   CPPFLAGS=$old_cppflags
   LDFLAGS=$old_ldflags
   LIBS=$old_libs
fi

# set preprocessor definition
if test "$MESA" != "no"; then
   AC_DEFINE(HAVE_MESA,[1])
   if test "$tryMangledMesa" = "yes"; then
      MESA_CPPFLAGS="$MESA_CPPFLAGS -DUSE_MGL_NAMESPACE";
   fi
else
   MESA_CPPFLAGS=""
   MESA_LDFLAGS=""
   MESA_LIBS=""
fi

AC_MSG_CHECKING(for Mesa)
AC_MSG_RESULT($MESA)
AC_SUBST(MESA)
AC_SUBST(MESA_CPPFLAGS)
AC_SUBST(MESA_LDFLAGS)
AC_SUBST(MESA_LIBS)



## ---------------------------------------------------------------------------
## Add some directories to search for includes and libraries
## ---------------------------------------------------------------------------

# Fill these out when we have something to put here.
# It just causes warnings until then....
#CPPFLAGS="-I$TOPDIR/common $CPPFLAGS"
#LDFLAGS="-L$TOPDIR/lib $LDFLAGS"

## ---------------------------------------------------------------------------
## Check some data type sizes
## ---------------------------------------------------------------------------
AC_CHECK_SIZEOF(int,0)
AC_CHECK_SIZEOF(long,0)
AC_CHECK_SIZEOF(long long,0)
AC_CHECK_SIZEOF(void *,0)

AC_C_BIGENDIAN

## ---------------------------------------------------------------------------
## Generate configuration files
## ---------------------------------------------------------------------------
if test "$CUDA" != "no"; then
   MAKETARGETSCUDA=config/make-targets-cuda
else
   MAKETARGETSCUDA=config/make-targets-nocuda
fi
AC_SUBST_FILE(MAKETARGETSCUDA)

AC_SUBST_FILE(DEPENDENCIES)
DEPENDENCIES=config/make-dependencies
AC_SUBST_FILE(VARIABLES)
VARIABLES=config/make-variables
AC_SUBST_FILE(TARGETS)
TARGETS=config/make-targets

## ---------------------------------------------------------------------------
## Do the standard flags and a few nonstandard ones
## ---------------------------------------------------------------------------
AC_SUBST(CPPFLAGS)
AC_SUBST(CXXFLAGS)
AC_SUBST(CFLAGS)
AC_SUBST(LDFLAGS)
AC_SUBST(LIBS)

## ---------------------------------------------------------------------------
## Subdirectories are separated by capability
## ---------------------------------------------------------------------------
SUBDIRS="$""(SUBDIRS_COMMON)"
if test "$MPI" != "no"; then
  SUBDIRS="$SUBDIRS $""(SUBDIRS_MPI)"
fi
if test "$CUDA" != "no"; then
  SUBDIRS="$SUBDIRS $""(SUBDIRS_CUDA)"
fi
AC_SUBST(SUBDIRS)


## ---------------------------------------------------------------------------
## Final makefile transformations; note order can be important here
## ---------------------------------------------------------------------------

AC_OUTPUT(config/make-dependencies
          config/make-variables
          config/make-targets-cuda
          config/make-targets-nocuda
          config/make-targets
          Makefile
          src/Makefile
          test/Makefile
)
