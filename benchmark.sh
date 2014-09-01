#!/bin/bash
ARCH="-cpu"

echo "# Dragon 100K"
test/testray -f test/dragon.obj -aa  -res 1080 1920 -cp 6 6  8  -cu 0 1 0 -cla 0 3 0 -lp 11 -1.1 6.33 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
test/testray -f test/dragon.obj -aa  -res 1080 1920 -cp 6 6  8  -cu 0 1 0 -cla 0 3 0 -lp 11 -1.1 6.33 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
test/testray -f test/dragon.obj -aa  -res 1080 1920 -cp 6 6  8  -cu 0 1 0 -cla 0 3 0 -lp 11 -1.1 6.33 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
#test/testray -f test/dragon.obj -aa  -res 1080 1920 -cp 6 6  8  -cu 0 1 0 -cla 0 3 0 -lp 11 -1.1 6.33 50 1 0 0 -fovx 45 -fovy 30 -o dragon1.bmp -ao 16


test/testray -f test/dragon.obj -aa  -res 1080 1920 -cp 1 2  3  -cu 0 1 0 -cla 0 3 0 -lp 11 -1.1 6.33 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
test/testray -f test/dragon.obj -aa  -res 1080 1920 -cp 1 2  3  -cu 0 1 0 -cla 0 3 0 -lp 11 -1.1 6.33 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
test/testray -f test/dragon.obj -aa  -res 1080 1920 -cp 1 2  3  -cu 0 1 0 -cla 0 3 0 -lp 11 -1.1 6.33 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH

#test/testray -f test/dragon.obj -aa  -res 1080 1920 -cp 1 2  3  -cu 0 1 0 -cla 0 3 0 -lp 11 -1.1 6.33 50 1 0 0 -fovx 45 -fovy 30 -o dragon2.bmp -ao 16

test/testray -f test/dragon.obj -aa  -res 1080 1920 -cp 2 0  8  -cu 0 1 0 -cla 0 3 0 -lp 11 -1.1 6.33 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
test/testray -f test/dragon.obj -aa  -res 1080 1920 -cp 2 0  8  -cu 0 1 0 -cla 0 3 0 -lp 11 -1.1 6.33 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
test/testray -f test/dragon.obj -aa  -res 1080 1920 -cp 2 0  8  -cu 0 1 0 -cla 0 3 0 -lp 11 -1.1 6.33 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
#test/testray -f test/dragon.obj -aa  -res 1080 1920 -cp 2 0  8  -cu 0 1 0 -cla 0 3 0 -lp 11 -1.1 6.33 50 1 0 0 -fovx 45 -fovy 30 -o dragon4.bmp -ao 8


test/testray -f test/dragon.obj -aa  -res 1080 1920 -cp -8.01 9.1  -3  -cu 0 1 0 -cla 0 5.5 2 -lp 11 -9  0 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
test/testray -f test/dragon.obj -aa  -res 1080 1920 -cp -8.01 9.1  -3  -cu 0 1 0 -cla 0 5.5 2 -lp 11 -9  0 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
test/testray -f test/dragon.obj -aa  -res 1080 1920 -cp -8.01 9.1  -3  -cu 0 1 0 -cla 0 5.5 2 -lp 11 -9  0 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
#test/testray -f test/dragon.obj -aa  -res 1080 1920 -cp -8.01 9.1  -3  -cu 0 1 0 -cla 0 5.5 2 -lp 11 -9  0 50 1 0 0 -fovx 45 -fovy 30 -o dragon4.bmp -ao 8




echo "# Conference Room"

test/testray -f test/conference.obj -aa  -res 1080 1920 -cp 1 7 6 -cu 0 0 1 -cla 10 8 5 -lp 1 7 6 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
test/testray -f test/conference.obj -aa  -res 1080 1920 -cp 1 7 6 -cu 0 0 1 -cla 10 8 5 -lp 1 7 6 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
test/testray -f test/conference.obj -aa  -res 1080 1920 -cp 1 7 6 -cu 0 0 1 -cla 10 8 5 -lp 1 7 6 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
#test/testray -f test/conference.obj -aa  -res 1080 1920 -cp 1 7 6 -cu 0 0 1 -cla 10 8 5 -lp 1 7 6 50 1 0 0 -fovx 45 -fovy 30 -o conf1.bmp -ao 16

test/testray -f test/conference.obj -aa  -res 1080 1920 -cp 18 3 7 -cu 0 0 1 -cla 22 8 5 -lp 1 7 6 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
test/testray -f test/conference.obj -aa  -res 1080 1920 -cp 18 3 7 -cu 0 0 1 -cla 22 8 5 -lp 1 7 6 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
test/testray -f test/conference.obj -aa  -res 1080 1920 -cp 18 3 7 -cu 0 0 1 -cla 22 8 5 -lp 1 7 6 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
#test/testray -f test/conference.obj -aa  -res 1080 1920 -cp 18 3 7 -cu 0 0 1 -cla 22 8 5 -lp 1 7 6 50 1 0 0 -fovx 45 -fovy 30 -o conf2.bmp -ao 16

test/testray -f test/conference.obj -aa  -res 1080 1920 -cp 18 9 7 -cu 0 0 1 -cla 11 2 3 -lp 1 7 6 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
test/testray -f test/conference.obj -aa  -res 1080 1920 -cp 18 9 7 -cu 0 0 1 -cla 11 2 3 -lp 1 7 6 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
test/testray -f test/conference.obj -aa  -res 1080 1920 -cp 18 9 7 -cu 0 0 1 -cla 11 2 3 -lp 1 7 6 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
#test/testray -f test/conference.obj -aa  -res 1080 1920 -cp 18 9 7 -cu 0 0 1 -cla 11 2 3 -lp 1 7 6 50 1 0 0 -fovx 45 -fovy 30 -o conf3.bmp -ao 16

test/testray -f test/conference.obj -aa  -res 1080 1920 -cp 18 9 7 -cu 0 0 1 -cla 25 2 3 -lp 1 7 6 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
test/testray -f test/conference.obj -aa  -res 1080 1920 -cp 18 9 7 -cu 0 0 1 -cla 25 2 3 -lp 1 7 6 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
test/testray -f test/conference.obj -aa  -res 1080 1920 -cp 18 9 7 -cu 0 0 1 -cla 25 2 3 -lp 1 7 6 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
#test/testray -f test/conference.obj -aa  -res 1080 1920 -cp 18 9 7 -cu 0 0 1 -cla 25 2 3 -lp 1 7 6 50 1 0 0 -fovx 45 -fovy 30 -o conf4.bmp -ao 16

echo "# Sponza"
test/testray -f test/sponza.obj -aa  -res 1080 1920 -cp 14 8.1 0.73 -cu 0 1 0 -cla .01 4 0 -lp 11 -1.1 6.33 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
test/testray -f test/sponza.obj -aa  -res 1080 1920 -cp 14 8.1 0.73 -cu 0 1 0 -cla .01 4 0 -lp 11 -1.1 6.33 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
test/testray -f test/sponza.obj -aa  -res 1080 1920 -cp 14 8.1 0.73 -cu 0 1 0 -cla .01 4 0 -lp 11 -1.1 6.33 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
#test/testray -f test/sponza.obj -aa  -res 1080 1920 -cp 14 8.1 0.73 -cu 0 1 0 -cla .01 4 0 -lp 11 -1.1 6.33 50 1 0 0 -fovx 45 -fovy 30 -o sponza1.bmp -ao 16

test/testray -f test/sponza.obj -aa  -res 1080 1920 -cp 12 8.1 0.73 -cu 0 1 0 -cla .01 4 0 -lp 11 -1.1 6.33 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
test/testray -f test/sponza.obj -aa  -res 1080 1920 -cp 12 8.1 0.73 -cu 0 1 0 -cla .01 4 0 -lp 11 -1.1 6.33 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
test/testray -f test/sponza.obj -aa  -res 1080 1920 -cp 12 8.1 0.73 -cu 0 1 0 -cla .01 4 0 -lp 11 -1.1 6.33 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
#test/testray -f test/sponza.obj -aa  -res 1080 1920 -cp 12 8.1 0.73 -cu 0 1 0 -cla .01 4 0 -lp 11 -1.1 6.33 50 1 0 0 -fovx 45 -fovy 30 -o sponza2.bmp -ao 16

test/testray -f test/sponza.obj -aa  -res 1080 1920 -cp 5 2 0.73 -cu 0 1 0 -cla .01 4 0 -lp 11 -1.1 6.33 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
test/testray -f test/sponza.obj -aa  -res 1080 1920 -cp 5 2 0.73 -cu 0 1 0 -cla .01 4 0 -lp 11 -1.1 6.33 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
test/testray -f test/sponza.obj -aa  -res 1080 1920 -cp 5 2 0.73 -cu 0 1 0 -cla .01 4 0 -lp 11 -1.1 6.33 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
#test/testray -f test/sponza.obj -aa  -res 1080 1920 -cp 5 2 0.73 -cu 0 1 0 -cla .01 4 0 -lp 11 -1.1 6.33 50 1 0 0 -fovx 45 -fovy 30 -o sponza3.bmp -ao 16

test/testray -f test/sponza.obj -aa  -res 1080 1920 -cp 5 8 3.73 -cu 0 1 0 -cla .01 4 0 -lp 11 -1.1 6.33 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
test/testray -f test/sponza.obj -aa  -res 1080 1920 -cp 5 8 3.73 -cu 0 1 0 -cla .01 4 0 -lp 11 -1.1 6.33 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
test/testray -f test/sponza.obj -aa  -res 1080 1920 -cp 5 8 3.73 -cu 0 1 0 -cla .01 4 0 -lp 11 -1.1 6.33 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
#test/testray -f test/sponza.obj -aa  -res 1080 1920 -cp 5 8 3.73 -cu 0 1 0 -cla .01 4 0 -lp 11 -1.1 6.33 50 1 0 0 -fovx 45 -fovy 30 -o sponza4.bmp -ao 16

echo "# Buddha"

test/testray -f test/buddha.obj -aa  -res 1080 1920 -cp .5 3.7 -4 -cu 0 1 0 -cla .01 3 0 -lp 11 1.1 -6.33 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
test/testray -f test/buddha.obj -aa  -res 1080 1920 -cp .5 3.7 -4 -cu 0 1 0 -cla .01 3 0 -lp 11 1.1 -6.33 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
test/testray -f test/buddha.obj -aa  -res 1080 1920 -cp .5 3.7 -4 -cu 0 1 0 -cla .01 3 0 -lp 11 1.1 -6.33 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
#test/testray -f test/buddha.obj -aa  -res 1080 1920 -cp .5 3.7 -4 -cu 0 1 0 -cla .01 3 0 -lp 11 1.1 -6.33 50 1 0 0 -fovx 45 -fovy 30 -o buddha1.bmp -ao 16

test/testray -f test/buddha.obj -aa  -res 1080 1920 -cp .5 3.7 -3 -cu 0 1 0 -cla .01 3 0 -lp 11 1.1 -6.33 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
test/testray -f test/buddha.obj -aa  -res 1080 1920 -cp .5 3.7 -3 -cu 0 1 0 -cla .01 3 0 -lp 11 1.1 -6.33 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
test/testray -f test/buddha.obj -aa  -res 1080 1920 -cp .5 3.7 -3 -cu 0 1 0 -cla .01 3 0 -lp 11 1.1 -6.33 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
#test/testray -f test/buddha.obj -aa  -res 1080 1920 -cp .5 3.7 -3 -cu 0 1 0 -cla .01 3 0 -lp 11 1.1 -6.33 50 1 0 0 -fovx 45 -fovy 30 -o buddha1.bmp -ao 16

test/testray -f test/buddha.obj -aa  -res 1080 1920 -cp .5 -3.7 -3 -cu 0 1 0 -cla .01 0 0 -lp 11 1.1 -6.33 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
test/testray -f test/buddha.obj -aa  -res 1080 1920 -cp .5 -3.7 -3 -cu 0 1 0 -cla .01 0 0 -lp 11 1.1 -6.33 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
test/testray -f test/buddha.obj -aa  -res 1080 1920 -cp .5 -3.7 -3 -cu 0 1 0 -cla .01 0 0 -lp 11 1.1 -6.33 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH

echo "# hairball no"
#test/testray -f test/hairball.obj -aa  -res 1080 1920 -cp 6 6 6 -cu 0 1 0 -cla 0 0 0 -lp 11 1.1 15 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
#test/testray -f test/hairball.obj -aa  -res 1080 1920 -cp 6 6 6 -cu 0 1 0 -cla 0 0 0 -lp 11 1.1 15 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
#test/testray -f test/hairball.obj -aa  -res 1080 1920 -cp 6 6 6 -cu 0 1 0 -cla 0 0 0 -lp 11 1.1 15 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH

#test/testray -f test/hairball.obj -aa  -res 1080 1920 -cp 4 4 4 -cu 0 1 0 -cla 0 0 0 -lp 11 1.1 15 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
#test/testray -f test/hairball.obj -aa  -res 1080 1920 -cp 4 4 4 -cu 0 1 0 -cla 0 0 0 -lp 11 1.1 15 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
#test/testray -f test/hairball.obj -aa  -res 1080 1920 -cp 4 4 4 -cu 0 1 0 -cla 0 0 0 -lp 11 1.1 15 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH

#test/testray -f test/hairball.obj -aa  -res 1080 1920 -cp 3 3 3 -cu 0 1 0 -cla 0 0 0 -lp 11 1.1 15 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
#test/testray -f test/hairball.obj -aa  -res 1080 1920 -cp 3 3 3 -cu 0 1 0 -cla 0 0 0 -lp 11 1.1 15 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
#test/testray -f test/hairball.obj -aa  -res 1080 1920 -cp 3 3 3 -cu 0 1 0 -cla 0 0 0 -lp 11 1.1 15 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH

 
echo "# RM350"
test/testray -f test/RM_340K.obj -aa  -res 1080 1920 -cp 150 150 150 -cu 0 0 1 -cla 0 0 0 -lp 150 150 150 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
test/testray -f test/RM_340K.obj -aa  -res 1080 1920 -cp 150 150 150 -cu 0 0 1 -cla 0 0 0 -lp 150 150 150 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
test/testray -f test/RM_340K.obj -aa  -res 1080 1920 -cp 150 150 150 -cu 0 0 1 -cla 0 0 0 -lp 150 150 150 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH

test/testray -f test/RM_340K.obj -aa  -res 1080 1920 -cp -10 -10 -10 -cu 0 0 1 -cla 0 0 0 -lp 150 150 150 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
test/testray -f test/RM_340K.obj -aa  -res 1080 1920 -cp -10 -10 -10 -cu 0 0 1 -cla 0 0 0 -lp 150 150 150 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
test/testray -f test/RM_340K.obj -aa  -res 1080 1920 -cp -10 -10 -10 -cu 0 0 1 -cla 0 0 0 -lp 150 150 150 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH

test/testray -f test/RM_340K.obj -aa  -res 1080 1920 -cp 60 60 60 -cu 0 0 1 -cla 0 50 50 -lp 150 150 150 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
test/testray -f test/RM_340K.obj -aa  -res 1080 1920 -cp 60 60 60 -cu 0 0 1 -cla 0 50 50 -lp 150 150 150 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
test/testray -f test/RM_340K.obj -aa  -res 1080 1920 -cp 60 60 60 -cu 0 0 1 -cla 0 50 50 -lp 150 150 150 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH

echo "# RM650"
test/testray -f test/RM_650K.obj -aa  -res 1080 1920 -cp 175 175 175 -cu 0 0 1 -cla 0 0 0 -lp 150 150 150 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
test/testray -f test/RM_650K.obj -aa  -res 1080 1920 -cp 175 175 175 -cu 0 0 1 -cla 0 0 0 -lp 150 150 150 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
test/testray -f test/RM_650K.obj -aa  -res 1080 1920 -cp 175 175 175 -cu 0 0 1 -cla 0 0 0 -lp 150 150 150 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH

test/testray -f test/RM_650K.obj -aa  -res 1080 1920 -cp -10  200 -50 -cu 0 0 -1 -cla 100 100 100  -lp 150 150 150 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
test/testray -f test/RM_650K.obj -aa  -res 1080 1920 -cp -10  200 -50 -cu 0 0 -1 -cla 100 100 100  -lp 150 150 150 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
test/testray -f test/RM_650K.obj -aa  -res 1080 1920 -cp -10  200 -50 -cu 0 0 -1 -cla 100 100 100  -lp 150 150 150 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH

test/testray -f test/RM_650K.obj -aa  -res 1080 1920 -cp 100  200 -50 -cu 0 0 -1 -cla 100 100 50  -lp 150 150 150 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
test/testray -f test/RM_650K.obj -aa  -res 1080 1920 -cp 100  200 -50 -cu 0 0 -1 -cla 100 100 50  -lp 150 150 150 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
test/testray -f test/RM_650K.obj -aa  -res 1080 1920 -cp 100  200 -50 -cu 0 0 -1 -cla 100 100 50  -lp 150 150 150 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH

echo "# RM_1M"
test/testray -f test/RM_1M.obj -aa  -res 1080 1920 -cp 100  200 -50 -cu 0 0 -1 -cla 100 100 50  -lp 150 150 150 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
test/testray -f test/RM_1M.obj -aa  -res 1080 1920 -cp 100  200 -50 -cu 0 0 -1 -cla 100 100 50  -lp 150 150 150 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
test/testray -f test/RM_1M.obj -aa  -res 1080 1920 -cp 100  200 -50 -cu 0 0 -1 -cla 100 100 50  -lp 150 150 150 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH

test/testray -f test/RM_1M.obj -aa  -res 1080 1920 -cp 143  -75 73 -cu 0 0 -1 -cla 100 100 50  -lp 150 150 150 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
test/testray -f test/RM_1M.obj -aa  -res 1080 1920 -cp 143  -75 73 -cu 0 0 -1 -cla 100 100 50  -lp 150 150 150 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
test/testray -f test/RM_1M.obj -aa  -res 1080 1920 -cp 143  -75 73 -cu 0 0 -1 -cla 100 100 50  -lp 150 150 150 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH

test/testray -f test/RM_1M.obj -aa  -res 1080 1920 -cp 126  -115 101 -cu 0 0 -1 -cla 100 100 50  -lp 150 150 150 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
test/testray -f test/RM_1M.obj -aa  -res 1080 1920 -cp 126  -115 101 -cu 0 0 -1 -cla 100 100 50  -lp 150 150 150 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
test/testray -f test/RM_1M.obj -aa  -res 1080 1920 -cp 126  -115 101 -cu 0 0 -1 -cla 100 100 50  -lp 150 150 150 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH

echo "# RM_1_7M "

test/testray -f test/RM_1.7M.obj -aa  -res 1080 1920 -cp 69  -63 122 -cu 0 0 -1 -cla 100 100 50  -lp 150 150 150 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
test/testray -f test/RM_1.7M.obj -aa  -res 1080 1920 -cp 69  -63 122 -cu 0 0 -1 -cla 100 100 50  -lp 150 150 150 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
test/testray -f test/RM_1.7M.obj -aa  -res 1080 1920 -cp 69  -63 122 -cu 0 0 -1 -cla 100 100 50  -lp 150 150 150 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH

test/testray -f test/RM_1.7M.obj -aa  -res 1080 1920 -cp 143  -75 73 -cu 0 0 -1 -cla 100 100 50  -lp 150 150 150 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
test/testray -f test/RM_1.7M.obj -aa  -res 1080 1920 -cp 143  -75 73 -cu 0 0 -1 -cla 100 100 50  -lp 150 150 150 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
test/testray -f test/RM_1.7M.obj -aa  -res 1080 1920 -cp 143  -75 73 -cu 0 0 -1 -cla 100 100 50  -lp 150 150 150 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH

test/testray -f test/RM_1.7M.obj -aa  -res 1080 1920 -cp 150  -150 150 -cu 0 0 -1 -cla 100 100 50  -lp 150 150 150 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
test/testray -f test/RM_1.7M.obj -aa  -res 1080 1920 -cp 150  -150 150 -cu 0 0 -1 -cla 100 100 50  -lp 150 150 150 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
test/testray -f test/RM_1.7M.obj -aa  -res 1080 1920 -cp 150  -150 150 -cu 0 0 -1 -cla 100 100 50  -lp 150 150 150 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH

echo "# RM_3_2M "
test/testray -f test/RM_3.2M.obj -aa  -res 1080 1920 -cp 150  -150 150 -cu 0 0 -1 -cla 200 200 100  -lp 150 150 150 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
test/testray -f test/RM_3.2M.obj -aa  -res 1080 1920 -cp 150  -150 150 -cu 0 0 -1 -cla 200 200 100  -lp 150 150 150 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH
test/testray -f test/RM_3.2M.obj -aa  -res 1080 1920 -cp 150  -150 150 -cu 0 0 -1 -cla 200 200 100  -lp 150 150 150 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH

test/testray -f test/RM_3.2M.obj -aa  -res 1080 1920 -cp 150  150 150 -cu 0 0 -1 -cla 200 200 100  -lp 150 150 150 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH 
test/testray -f test/RM_3.2M.obj -aa  -res 1080 1920 -cp 150  150 150 -cu 0 0 -1 -cla 200 200 100  -lp 150 150 150 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH 
test/testray -f test/RM_3.2M.obj -aa  -res 1080 1920 -cp 150  150 150 -cu 0 0 -1 -cla 200 200 100  -lp 150 150 150 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH 

test/testray -f test/RM_3.2M.obj -aa  -res 1080 1920 -cp 650  350 150 -cu 0 0 -1 -cla 200 200 100  -lp 150 150 150 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH 
test/testray -f test/RM_3.2M.obj -aa  -res 1080 1920 -cp 650  350 150 -cu 0 0 -1 -cla 200 200 100  -lp 150 150 150 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH 
test/testray -f test/RM_3.2M.obj -aa  -res 1080 1920 -cp 650  350 150 -cu 0 0 -1 -cla 200 200 100  -lp 150 150 150 50 1 0 0 -fovx 45 -fovy 30 -test 50 100 $ARCH 


#test/testray -f test/CornellBox-Original.obj -aa  -res 1080 1920 -cp .8 .8 3 -cu 0 1 0 -cla  0 1 0  -lp  0 1.97 0  50 1 0 0 -fovx 45 -fovy 30 -ao 32 1
