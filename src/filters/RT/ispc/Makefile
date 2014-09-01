
CXX=g++
CXXFLAGS= -O3 -Wall
ISPC=ispc
ISPCFLAGS=-O2 --arch=x86-64 --target=avx1.1 --cpu=corei7-avx


.PHONY: clean
.PRECIOUS: rt.ispc

librtispc.a: rt_ispc.o  
	ar rcs $@ $^

rt_ispc.o: rt.ispc
	$(ISPC) $(ISPCFLAGS) rt.ispc -o rt_ispc.o -h rt_ispc.h

clean:
	rm -f lib*
	rm -f *ispc.h
	rm -f *ispc.o