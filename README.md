# Using python

Just using python file (NOT C++)

``` bash
python3 stereovison_disparity.py
```

# BUILD with CMake

## prerequisite

* cmake
* fmt
* jsoncpp
* yaml-cpp

#### Ubuntu

``` bash
sudo apt-get install cmake libfmt-dev libjsoncpp-dev libyaml-cpp-dev
```

#### MacOS

``` bash
brew install cmake fmt jsoncpp yaml-cpp
```

#### Windows

``` bash
NOT YET TESTED
```

## Build

``` bash
mkdir build && cd build && cmake .. && make -j2
```

## Run

In build file which is made above(build section).

``` bash
./stereovision_disparity
```

## Test source, Switch between StereoBM and StereoSGBM

In CamModify.h file

``` c++
   /* ----------- CamModify.h -------------- */
 9 // line number  10
10 #define TEST 1
11 #define STEREOSGBM 1
   /** TEST
    * if TEST is set to 1, we gonna use pre recorded video in "data" directory, left.mp4, right.mp4.
    * By the way, set to 0, we will use connected video cam (But only compatible with ubuntu-linux)
    * /
   /** STEREOSGBM
    * if #define STEREOSGBM is set to 1, then we gonna use StereoSGBM.
    * However, it is set to 0, use StereoBM
    * /
```