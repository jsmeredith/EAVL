#! /bin/sh

if test -h "eavl"; then
   echo "Error: a symbolic link named 'eavl' already exists.  Aborting."
   exit 1
fi
if test -d "eavl"; then
   echo "Error: 'eavl' directory already exists.  Aborting."
   exit 1
fi
if test -e "eavl"; then
   echo "Error: a file named 'eavl' already exists.  Aborting."
   exit 1
fi

mkdir eavl
if test $? -ne 0; then
   echo "Error creating directory 'eavl'.  Aborting."
   exit 1
fi
if test ! -d "eavl"; then
   echo "Error: couldn't create directory 'eavl'.  Aborting."
   exit 1
fi

mkdir eavl/include
mkdir eavl/lib
cp src/*/*.h eavl/include/
cp config/*.h eavl/include/
cp lib/libeavl.* eavl/lib/
day=`date '+%Y-%m-%d'`
fn=eavl-$day.tar.gz
tar zcf $fn eavl/

if test $? -ne 0; then
   echo "Error creating tar file '$fn'.  Aborting."
   rm -rf eavl/
   exit 1
fi
if test ! -e $fn; then
   echo "Error: couldn't create tar file.  Aborting."
   rm -rf eavl/
   exit 1
fi

rm -rf eavl/
echo "Created package $fn"
