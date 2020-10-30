#ifndef MAINLOOP_HPP
#define MAINLOOP_HPP

#include "display.hpp"
#include "kernel.cuh"

#define c_window_width  1900
#define c_window_height 1600

#define c_world_width 1400
#define c_world_height 800


//using u8 = unsigned char;

#include "iostream"
using namespace std;

class MainLoop
{
public:
    MainLoop();

    Display display;


    parameter_set params;
    //GPU/CPU interop memory pointers:
    unsigned char state = 0;
    lbm_node* array1;
    lbm_node* array2;
    lbm_node* array1_gpu;
    lbm_node* array2_gpu;
    unsigned char* barrier;
    unsigned char* barrier_gpu;
    d2q9_node* d2q9_gpu;
    parameter_set* params_gpu;

    char needsUpdate = 1;
    int prex = -1;
    int prey = -1;

    uchar4* image;


    void Run();
    bool Init();

    void updatePixelsCPU(unsigned char* data);
    int number_of_colors=512;
    //std::vector<sf::Color> sf_color_list;
    std::vector<dev_color> host_color_list;
    dev_color *dev_color_list;

    // Array of generated pixels on device.
    unsigned char *devData;
    // Array to copy generated pixels to on host.
    unsigned char *hostData;
    // Allocate host and device memory


    //cuda error variables:
    cudaError_t ierrAsync;
    cudaError_t ierrSync;


    size_t N;
    void getParams(parameter_set* params)
    {
        params->viscosity = 0.005f;
        params->contrast = 1;
        params->v = 0.185f;
        params->mode = mCurl;
        params->width = display.tex_width;
        params->height =display.tex_height;
        params->stepsPerRender =25;

        params->zoom_u=1450;
        params->zoom_rho=10;
        params->zoom_curl=700;

        params->v3=params->v*3.0f;
        params->vquart= 3.0f*params->v*params->v;
        params->vquarth=1.5f*params->v*params->v;
    }




    //--------------------------------------------------------------------------------//
    //                        CUDA INITIALIZER FUNCTIONS                              //
    //--------------------------------------------------------------------------------//

    //provide LBM constants for d2q9 style nodes
    //assumes positive is up and right, whereas our program assumes positive down and right.
    void init_d2q9(d2q9_node* d2q9);
    void zeroSite(lbm_node* array, int index);
    void initBoundaries();
    void initFluid();
     void initFluid_flow();

    //determine front and back lattice buffer orientation
    //and launch all 3 LBM kernels
    void kernelLauncher(unsigned char* image, bool& canDraw);

    //-----------------------------------------------------------//
    //                  BARRIER FUNCTIONS                        //
    //-----------------------------------------------------------//

    void clearBarriers();
    void drawLineDiagonal();
    void drawSquare();
    void loadBarrierFromFile();
    void FreePointers();

    //------------------------------------------------------------------------------//
    //                                HELPER FUNCTIONS                              //
    //------------------------------------------------------------------------------//

    //get 1d flat index from row and col
    int getIndex_cpu(int x, int y) {
        return y * params.width + x;
    }

    //display stats of all detected cuda capable devices,
    //and return the number
    int deviceQuery()
    {
        cudaDeviceProp prop;
        int nDevices = 1;
        cudaError_t ierr;


        ierr = cudaGetDeviceCount(&nDevices);

        int i = 0;
        for (i = 0; i < nDevices; ++i)
        {
            ierr = cudaGetDeviceProperties(&prop, i);
            printf("Device number: %d\n", i);
            printf("  Device name: %s\n", prop.name);
            printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
            printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
            printf("  Max threads in X-dimension of block: %d\n", prop.maxThreadsDim[0]);
            printf("  Max threads in Y-dimension of block: %d\n", prop.maxThreadsDim[1]);
            printf("  Max threads in Z-dimension of block: %d\n\n", prop.maxThreadsDim[2]);
            if (ierr != cudaSuccess) { printf("error: %s\n", cudaGetErrorString(ierr)); }
        }

        return nDevices;
    }



    int RandomInt(int min, int max)
    {
        return std::rand()%((max - min) + 1) + min;
    }


};

#endif // MAINLOOP_HPP
