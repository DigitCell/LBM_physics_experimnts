#include "kernel.cuh"






bool step( parameter_set &params, parameter_set* params_gpu, d2q9_node* d2q9_gpu,
           lbm_node* array1_gpu,lbm_node* array2_gpu,unsigned char* barrier_gpu,
           int &prex,  int &prey, unsigned char* image,
           cudaError_t &ierrAsync, cudaError_t &ierrSync, bool& updateDraw,dev_color *dev_color, int zoom)
{

    lbm_node* before = array1_gpu;
    lbm_node* after = array2_gpu;

    DEBUG_PRINT(("these are the addresses: \n\t\tb4=%p\taft=%p\n\t\tar1=%p\tar2=%p", before, after, array1_gpu, array2_gpu));
    //determine number of threads and blocks required
    dim3 threads_per_block = dim3(32, 32, 1);
    dim3 number_of_blocks = dim3(params.width / 32 + 1, params.height / 32 + 1, 1);

    collide << < number_of_blocks, threads_per_block >> > (d2q9_gpu, before, after, params_gpu, barrier_gpu);

    ierrSync = cudaGetLastError();
    ierrAsync = cudaDeviceSynchronize(); // Wait for the GPU to finish
    if (ierrSync != cudaSuccess) { DEBUG_PRINT(("Sync error: %s\n", cudaGetErrorString(ierrSync))); }
    if (ierrAsync != cudaSuccess) { DEBUG_PRINT(("Async error: %s\n", cudaGetErrorString(ierrAsync))); }

    before = array2_gpu;
    after = array1_gpu;
    stream << < number_of_blocks, threads_per_block >> > (d2q9_gpu, before, after, barrier_gpu, params_gpu);

    ierrSync = cudaGetLastError();
    ierrAsync = cudaDeviceSynchronize(); // Wait for the GPU to finish
    if (ierrSync != cudaSuccess) { DEBUG_PRINT(("Sync error: %s\n", cudaGetErrorString(ierrSync))); }
    if (ierrAsync != cudaSuccess) { DEBUG_PRINT(("Async error: %s\n", cudaGetErrorString(ierrAsync))); }


    bounce << < number_of_blocks, threads_per_block >> > (d2q9_gpu, before, after, barrier_gpu, params_gpu, image, prex, prey);

    ierrSync = cudaGetLastError();
    ierrAsync = cudaDeviceSynchronize(); // Wait for the GPU to finish
    if (ierrSync != cudaSuccess) { DEBUG_PRINT(("Sync error: %s\n", cudaGetErrorString(ierrSync))); }
    if (ierrAsync != cudaSuccess) { DEBUG_PRINT(("Async error: %s\n", cudaGetErrorString(ierrAsync))); }

    if(updateDraw)
    {
        render << < number_of_blocks, threads_per_block >> > (d2q9_gpu, before, after, barrier_gpu, params_gpu, image,
                                                              prex, prey,dev_color, zoom);

        ierrSync = cudaGetLastError();
        ierrAsync = cudaDeviceSynchronize(); // Wait for the GPU to finish
        if (ierrSync != cudaSuccess) { DEBUG_PRINT(("Sync error: %s\n", cudaGetErrorString(ierrSync))); }
        if (ierrAsync != cudaSuccess) { DEBUG_PRINT(("Async error: %s\n", cudaGetErrorString(ierrAsync))); }
    }

   // { DEBUG_PRINT(("Kernel work normal", cudaGetErrorString(ierrSync))); }

    return true;
}



//--------------------------------------------------------------------------------//
//                   CUDA HELPER AND RENDER FUNCTIONS                             //
//--------------------------------------------------------------------------------//
__device__
unsigned char clip(int n) {
    return n > 255 ? 255 : (n < 0 ? 0 : n);
}

__device__
int clipInt(float n) {
    if(abs(n)>254)
        return 255;
    else
        return (int)(abs(n));

   // return n > 255 ? 255 : (n < 0 ? 0 : n);
}

//get 1d flat index from row and col
__device__
int getIndex(int x, int y, parameter_set* params)
{
    return y * params->width + x;
}

__device__
void printNode(lbm_node* node, lbm_node* before, lbm_node* after)
{
    DEBUG_PRINT(("\t\t\ttest: %x\n", after));
    DEBUG_PRINT(("\trho: %.6f\n\tux: %.6f\n\tuy: %.6f\n\tvN: %.6f\n\tvE: %.6f\n\tvW: %.6f\n\tvS: %.6f\n\tv0: %.6f\n\tvNW: %.6f\n\tvNE: %.6f\n\tvSW: %.6f\n\tvSE: %.6f\n",
        node->rho,
        node->ux,
        node->uy,
        (node->f)[dN],
        (node->f)[dE],
        (node->f)[dW],
        (node->f)[dS],
        (node->f)[d0],
        (node->f)[dNW],
        (node->f)[dNE],
        (node->f)[dSW],
        (node->f)[dSW]
        ));

    DEBUG_PRINT(("\n\tbefore: %p \n\tafter: %p \n\t node : %p \n", before, after, node));
}

__device__
uchar4 getRGB_roh(float irho, parameter_set* params)
{
    uchar4 val;

    if (irho == irho)
    {
        int jrho= (1.0f - irho) * 255 * params->zoom_rho; // approximately -255 to 255;

        val.x = 0;
        val.w = 255;
        val.z = 255;

        if (jrho > 0)
        {
            val.y =clip(jrho);
            val.z = 255;
        }
        else
        {
            val.z =clip(-jrho);
            val.y = 0;
        }
    }
    else
    {
        val.y = 0;
        val.x = 255;
        val.w = 0;
        val.z = 255;
    }

     return val;


}

__device__
uchar4 getRGB_u(float i, parameter_set* params)
{

    uchar4 val;
    if (i == i)
    {
        val.w = 255;
        val.x = 0;
        val.y = 0;
        val.z = clip(i*params->zoom_u / 1.0);
    }
    else
    {
        val.w = 255;
        val.x = 255;
        val.y = 0;
        val.z = 0;
    }
    return val;
}

__device__
float computeCurlMiddleCase(int x, int y, lbm_node * array1, parameter_set* params) {
    return (array1[getIndex(x, y + 1, params)].ux
        - array1[getIndex(x, y - 1, params)].ux)
        - (array1[getIndex(x + 1, y, params)].uy
            - array1[getIndex(x - 1, y, params)].uy);
}

__device__
uchar4 getRGB_curl(int x, int y, lbm_node* array, parameter_set* params, dev_color *dev_color, int zoom)
{

    uchar4 val;
    val.x = 0;
    val.y = 0;
    val.z = 0;
    val.w = 255;


    if (0 < x && x < params->width - 1) {
        if (0 < y && y < params->height - 1) {
            //picture[getIndex(x,y)]
            float computeCurl=computeCurlMiddleCase(x, y, array, params);

            if (computeCurl>0 )
            {
                if(abs(computeCurl)<1)
                {
                    auto tempColor=dev_color[clipInt( params->zoom_curl * sqrt(computeCurl))];
                    val.x = tempColor.r;
                    val.y = tempColor.g;
                    val.z = tempColor.b;
                    val.w=255;
                }
                else
                {
                    auto tempColor=dev_color[clipInt(params->zoom_curl * computeCurl)];
                    val.x = tempColor.r;
                    val.y = tempColor.g;
                    val.z = tempColor.b;
                    val.w=255;
                }
            }
            else
            {
                if(abs(computeCurl)<1)
                {
                    auto tempColor=dev_color[255+clipInt(params->zoom_curl * sqrt(abs(computeCurl)))];
                    val.x = tempColor.r;
                    val.y = tempColor.g;
                    val.z = tempColor.b;
                    val.w=255;
                }
                else
                {
                    auto tempColor=dev_color[255+clipInt(params->zoom_curl * computeCurl)];
                    val.x = tempColor.r;
                    val.y = tempColor.g;
                    val.z = tempColor.b;
                    val.w=255;
                }
            }
        }
        //else {
        //	//picture[getIndex(x,y)]
        //	colorIndex = (int)(nColors * (0.5 + computeCurlEdgeCase(col, row, array) * contrast * 0.3));
        //}
    }

    if (array[getIndex(x, y, params)].rho != array[getIndex(x, y, params)].rho)
    {
        val.x = 255;
        val.y = 0;
        val.z = 0;
        val.w = 255;
    }

    return val;

}

__device__
void computeColor(lbm_node* array, int x, int y, parameter_set* params, unsigned char* image,
                  unsigned char* barrier, int prex, int prey,dev_color *dev_color, int zoom)
{
    int i = getIndex(x, y, params);
    int prei = getIndex(prex, prey, params);

     const auto index = 4 * (i);
    if (barrier[i] == 1)
    {
        image[index + 0]=  75;
        image[index + 1] = 75;
        image[index + 2] = 75;
        image[index + 3] = 75;
    }
    else
    {
        switch (params->mode)
        {
            case(mRho):
            {
                uchar4 itemp= getRGB_roh(array[i].rho, params);
                image[index + 0] =  itemp.x;
                image[index + 1] =  itemp.y;
                image[index + 2] =  itemp.z;
                image[index + 3] =  itemp.w;
                break;
            }
            case(mCurl):
            {
                uchar4 itemp= getRGB_curl(x, y, array, params, dev_color, zoom);
                image[index + 0] =  itemp.x;
                image[index + 1] =  itemp.y;
                image[index + 2] =  itemp.z;
                image[index + 3] =  itemp.w;
                break;
            }

            case(mSpeed):
            {
                 uchar4 itemp= getRGB_u(sqrt(array[i].ux * array[i].ux + array[i].uy * array[i].uy), params);
                 image[index + 0] =  itemp.x;
                 image[index + 1] =  itemp.y;
                 image[index + 2] =  itemp.z;
                 image[index + 3] =  itemp.w;
                break;
            }

            /*
        case(mUx):
            image[i] = getRGB_u(array[i].ux);
            break;
        case(mUy):
            image[i] = getRGB_u(array[i].uy);
            break;
            */
        }
    }
    if (i == prei)
    {
        image[index + 0]=  255;
        image[index + 1] = 0;
        image[index + 2] = 0;
        image[index + 3] = 255;
    }
}

//--------------------------------------------------------------------------------//
//                   CUDA COLLIDE STEP KERNEL AND DEVICES                         //
//--------------------------------------------------------------------------------//

__device__
void macro_gen(float* f, float* ux, float* uy, float* rho, int i, parameter_set* params)
{
    const float top_row = f[6] + f[2] + f[5];
    const float mid_row = f[3] + f[0] + f[1];
    const float bot_row = f[7] + f[4] + f[8];

    if (i == getIndex(trace_x, trace_y, params))
        for (int i = 0; i < 9;i++)
        {
            DEBUG_PRINT(("\t\tmacro_gen: f[%d]=%.6f\n", i, f[i]));
        }

    *rho = top_row + mid_row + bot_row;
    if (*rho > 0)
    {
        *ux = ((f[5] + f[1] + f[8]) - (f[6] + f[3] + f[7])) / (*rho);
        *uy = (bot_row - top_row) / (*rho);
    }
    else
    {
        *ux = 0;
        *uy = 0;
    }

    return;
}

//return acceleration
__device__
float accel_gen(int node_num, float ux, float uy, float u2, float rho, d2q9_node* d2q9)
{
    float u_direct = ux * d2q9[node_num].ex + uy * (-d2q9[node_num].ey);
    float unweighted = 1 + 3 * u_direct + 4.5*u_direct*u_direct - 1.5*u2;

    return rho * d2q9[node_num].wt * unweighted;
}

__global__
void collide(d2q9_node* d2q9, lbm_node* before, lbm_node* after, parameter_set* params, unsigned char* barrier)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int i = getIndex(x, y, params);


    float omega = 1.0f / (3.0f * params->viscosity + 0.5f);

    if(barrier[i]==2)
    {
        omega = 1.0f / (3.0f *params->viscosity + 0.5f);
    }

    //toss out out of bounds
    if (x<0 || x >= params->width || y<0 || y >= params->height)
        return;

    if (x == trace_x && y == trace_y)
    {
        DEBUG_PRINT(("\n\nPre-Collision (before):\n"));
        printNode(&(before[i]), before, after);
        DEBUG_PRINT(("\n\nPre-Collision (after) (not used):\n"));
        printNode(&(after[i]), before, after);
    }


    macro_gen(before[i].f, &(after[i].ux), &(after[i].uy), &(after[i].rho), i, params);

    //int dir = 0;


    for (int dir = 0; dir<9; dir += 1)
    {
        (after[i].f)[dir] = ( before[i].f)[dir]
                            + omega* (accel_gen(dir, after[i].ux, after[i].uy,
                              after[i].ux * after[i].ux
                            + after[i].uy * after[i].uy, after[i].rho, d2q9)
                            - (before[i].f)[dir]);
    }
    return;
}


//--------------------------------------------------------------------------------//
//                   CUDA STREAM STEP KERNEL AND DEVICES                          //
//--------------------------------------------------------------------------------//
__device__
void doLeftWall(int x, int y, lbm_node* after, d2q9_node* d2q9, float v, parameter_set* params)
{
    //DEBUG_PRINT(("setting left wall to %.6f (wt: %.3f, v: %.3f)\n", d2q9[dE].wt  * (1 + 3 * v + 3 * v * v), d2q9[dE].wt,v));
    float v3=params->v3;
    float vquart=params->vquart;
    float vquarth=params->vquarth;
    (after[getIndex(x, y, params)].f)[dE] =  d2q9[dE].wt  * (1 + v3 + vquart);
    (after[getIndex(x, y, params)].f)[dNE] = d2q9[dNE].wt * (1 + v3 + vquart);
    (after[getIndex(x, y, params)].f)[dSE] = d2q9[dSE].wt * (1 + v3 + vquart);
}

__device__
void doRightWall(int x, int y, lbm_node* after, d2q9_node* d2q9, float v, parameter_set* params)
{
    float v3=params->v3;
    float vquart=params->vquart;
    float vquarth=params->vquarth;
    (after[getIndex(x, y, params)].f)[dW] =  d2q9[dW].wt  * (1 - v3 + vquart);
    (after[getIndex(x, y, params)].f)[dNW] = d2q9[dNW].wt * (1 - v3 + vquart);
    (after[getIndex(x, y, params)].f)[dSW] = d2q9[dSW].wt * (1 - v3 + vquart);
}

//(top and bottom walls)
__device__
void doFlanks(int x, int y, lbm_node* after, d2q9_node* d2q9, float v, parameter_set* params)
{
    float v3=params->v3;
    float vquart=params->vquart;
    float vquarth=params->vquarth;
    (after[getIndex(x, y, params)].f)[d0] =  d2q9[d0].wt  * (1 - vquarth);
    (after[getIndex(x, y, params)].f)[dE] =  d2q9[dE].wt  * (1 + v3 + vquart);
    (after[getIndex(x, y, params)].f)[dW] =  d2q9[dW].wt  * (1 - v3 + vquart);
    (after[getIndex(x, y, params)].f)[dN] =  d2q9[dN].wt  * (1 - vquarth);
    (after[getIndex(x, y, params)].f)[dS] =  d2q9[dS].wt  * (1 - vquarth);
    (after[getIndex(x, y, params)].f)[dNE] = d2q9[dNE].wt * (1 + v3 + vquart);
    (after[getIndex(x, y, params)].f)[dSE] = d2q9[dSE].wt * (1 + v3 + vquart);
    (after[getIndex(x, y, params)].f)[dNW] = d2q9[dNW].wt * (1 - v3 + vquart);
    (after[getIndex(x, y, params)].f)[dSW] = d2q9[dSW].wt * (1 - v3 + vquart);
}

__device__
void streamEdgeCases(int x, int y, lbm_node* after, unsigned char* barrier,
    parameter_set* params, d2q9_node* d2q9)
{

    if (x == 0)
    {
        if (barrier[getIndex(x, y, params)] != 1)
        {
            //DEBUG_PRINT(("doing left wall!"));
            doLeftWall(x, y, after, d2q9, params->v, params);
        }
    }
    else if (x == params->width - 1)
    {
        if (barrier[getIndex(x, y, params)] != 1)
        {
            doRightWall(x, y, after, d2q9, params->v, params);
        }
    }
    else if (y == 0 || y == params->width - 1)
    {
        if (barrier[getIndex(x, y, params)] != 1)
        {
            doFlanks(x, y, after, d2q9, params->v, params);
        }
    }
}

//stream: handle particle propagation, ignoring edge cases.
__global__
void stream(d2q9_node* d2q9, lbm_node* before, lbm_node* after,
    unsigned char* barrier, parameter_set* params)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int i = getIndex(x, y, params);


    if (x == trace_x && y == trace_y)
    {
        DEBUG_PRINT(("\n\nPre-stream: (before)\n"));
        printNode(&(before[i]), before, after);
    }

    //toss out out of bounds and edge cases
    if (x < 0 || x >= params->width || y < 0 || y >= params->height)
        return;



    after[i].rho = before[i].rho;
    after[i].ux = before[i].ux;
    after[i].uy = before[i].uy;

    if (!(x > 0 && x < params->width - 1 && y > 0 && y < params->height - 1))
    {
        //return;
        streamEdgeCases(x, y, after, barrier, params, d2q9);
    }
    else
    {
        //propagate all f values around a bit

        for (int dir = 0; dir<9; dir++)
        {
            (after[getIndex(d2q9[dir].ex + x, -d2q9[dir].ey + y, params)].f)[dir] =
                before[i].f[dir];
        }
    }

}

//--------------------------------------------------------------------------------//
//                   CUDA BOUNCE STEP KERNEL AND DEVICES                          //
//--------------------------------------------------------------------------------//

/*__device__
void bounceEdgeCases(int x, int y, lbm_node* after, unsigned char* barrier,
parameter_set* params, d2q9_node* d2q9)
{

}*/

__global__
void bounce(d2q9_node* d2q9, lbm_node* before, lbm_node* after,
    unsigned char* barrier, parameter_set* params, unsigned char* image, int prex, int prey)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int i = getIndex(x, y, params);

    if (x == trace_x && y == trace_y)
    {
        DEBUG_PRINT(("\n\npre-barriers:\n"));
        printNode(&(after[i]), before, after);
    }

    //toss out out of bounds and edge cases
    if (x < 0 || x >= params->width || y < 0 || y >= params->height)
        return;

    if (x > 0 && x < params->width - 1 && y>0 && y < params->height - 1)
    {
        if (barrier[i] == 1)
        {
            //int dir;
            for (int dir = 1; dir < 9; dir += 1) //-------------- ?
            {
                if (d2q9[dir].op > 0 && (after[i].f)[dir]>0)
                {
                    (after[getIndex(d2q9[dir].ex + x, -d2q9[dir].ey + y, params)].f)[dir]
                        = (before[i].f)[d2q9[dir].op];
                    //printf("doin a barrier bounce! %d\n",dir);
                    //(after[i].f)[dir] += (after[i].f)[d2q9[dir].op];
                    //    + (after[i].f)[dir];
                    //(after[i].f)[d2q9[dir].op] = 0;

                }
            }
        }



    }
    else
    {
        //bounceEdgeCases(x, y, after, barrier, params, d2q9);
    }

    if (x == trace_x && y == trace_y)
    {
        DEBUG_PRINT(("\n\nFinal rendered:\n"));
        printNode(&(after[i]), before, after);
    }

    // computeColor(after, x, y, params, image, barrier, prex, prey);

}

__global__
void render(d2q9_node* d2q9, lbm_node* before, lbm_node* after,
    unsigned char* barrier, parameter_set* params, unsigned char* image, int prex, int prey,dev_color *dev_color, int zoom)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int i = getIndex(x, y, params);

    if (x < 0 || x >= params->width || y < 0 || y >= params->height)
        return;

    computeColor(after, x, y, params, image, barrier, prex, prey, dev_color, zoom);
}
