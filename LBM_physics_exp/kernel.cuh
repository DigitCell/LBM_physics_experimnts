#pragma once

#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "math.h"
#include "time.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"



#include "dependencies/common/inc/helper_cuda.h"
#include "dependencies/common/inc/helper_functions.h"

//for __syncthreads()
#ifndef __CUDACC__
#define __CUDACC__
#endif


#define trace_x 50
#define trace_y 57
#define DEBUG_DELAY 0

#ifdef DEBUG
# define DEBUG_PRINT(x) printf x
#else
# define DEBUG_PRINT(x) do {} while (0)
#endif

//#define NX 1200
//#define NY 400

//#include <device_functions.h>

enum directions {
    d0 = 0,
    dE,
    dN,
    dW,
    dS,
    dNE,
    dNW,
    dSW,
    dSE
};

enum render_modes {
    mRho=0,
    mCurl=1,
    mSpeed=2,
    mUx=3,
    mUy=4
};


struct dev_color{
        __host__ __device__ dev_color()
        {
            r=0;
            g=0;
            b=0;
        };
        __host__ __device__ dev_color(int _x, int _y, int _z)
        {
            r=_x;
            g=_y;
            b=_z;
        };

        int r=0;
        int g=0;
        int b=0;
};


struct lbm_node{
    //velocities:
    float ux;	//x velocity
    float uy;	//y velocity

    float rho;	//density. aka rho
    float f[9];
};


struct d2q9_node{
    char ex; //x location
    char ey; //y location
    float wt; //weight
    unsigned char op; //opposite char
};

struct parameter_set{
    float viscosity;
    float omega;
    unsigned int width=100;
    unsigned int height=100;
    float contrast;
    float v;
    float v3;
    float vquart;
    float vquarth;

    unsigned char mode;
    int stepsPerRender;

    int zoom_rho=1;
    int zoom_u=1;
    int zoom_curl=1;


    bool need_update=false;
};



// CUDA HELPER AND RENDER FUNCTIONS

__device__
unsigned char clip(int n);
__device__
int clipInt(float n);
//get 1d flat index from row and col
__device__
int getIndex(int x, int y, parameter_set* params);
__device__
void printNode(lbm_node* node, lbm_node* before, lbm_node* after);
__device__
uchar4 getRGB_roh(float i, parameter_set* params);
__device__
uchar4 getRGB_u(float i, parameter_set* params);
__device__
float computeCurlMiddleCase(int x, int y, lbm_node * array1, parameter_set* params);
__device__
uchar4 getRGB_curl(int x, int y, lbm_node* array, parameter_set* params, dev_color *dev_color, int zoom);
__device__
void computeColor(lbm_node* array, int x, int y, parameter_set* params, unsigned char* image,
                  unsigned char* barrier, int prex, int prey, dev_color *dev_color, int zoom);


// CUDA COLLIDE STEP KERNEL AND DEVICES

__device__
void macro_gen(float* f, float* ux, float* uy, float* rho, int i, parameter_set* params);
//return acceleration
__device__
float accel_gen(int node_num, float ux, float uy, float u2, float rho, d2q9_node* d2q9);
__global__
void collide(d2q9_node* d2q9, lbm_node* before, lbm_node* after, parameter_set* params, unsigned char* barrier);



//  CUDA STREAM STEP KERNEL AND DEVICES

__device__
void doLeftWall(int x, int y, lbm_node* after, d2q9_node* d2q9, float v, parameter_set* params);
__device__
void doRightWall(int x, int y, lbm_node* after, d2q9_node* d2q9, float v, parameter_set* params);
//(top and bottom walls)
__device__
void doFlanks(int x, int y, lbm_node* after, d2q9_node* d2q9, float v, parameter_set* params);
__device__
void streamEdgeCases(int x, int y, lbm_node* after, unsigned char* barrier,
    parameter_set* params, d2q9_node* d2q9);
//stream: handle particle propagation, ignoring edge cases.
__global__
void stream(d2q9_node* d2q9, lbm_node* before, lbm_node* after,
    unsigned char* barrier, parameter_set* params);


// CUDA BOUNCE STEP KERNEL AND DEVICES

/*__device__
void bounceEdgeCases(int x, int y, lbm_node* after, unsigned char* barrier,
parameter_set* params, d2q9_node* d2q9)
{

}*/
__global__
void bounce(d2q9_node* d2q9, lbm_node* before, lbm_node* after,
    unsigned char* barrier, parameter_set* params, unsigned char* image, int prex, int prey);

__global__
void render(d2q9_node* d2q9, lbm_node* before, lbm_node* after,
    unsigned char* barrier, parameter_set* params, unsigned char* image, int prex, int prey, dev_color *dev_color, int zoom);

bool step( parameter_set &params, parameter_set* params_gpu, d2q9_node* d2q9_gpu,
           lbm_node* array1_gpu,lbm_node* array2_gpu,unsigned char* barrier_gpu,
           int &prex,  int &prey, unsigned char* image,
           cudaError_t &ierrAsync, cudaError_t &ierrSync, bool& updateDraw, dev_color *dev_color, int zoom);



