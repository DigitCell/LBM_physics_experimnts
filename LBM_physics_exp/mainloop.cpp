#include "mainloop.hpp"
#include "mainloop.hpp"

MainLoop::MainLoop():
    display(c_window_width, c_window_height)

{
    std::cout<<"Start LBM";
    std::srand(45073);

    int half_number_colors=number_of_colors/2;

    for (int i=0; i<255; i++)
    {
        const tinycolormap::Color color = tinycolormap::GetColor((float(i))/half_number_colors, tinycolormap::ColormapType::Inferno);
        float rc=color.r()*255;
        float gc=color.g()*255;
        float bc=color.b()*255;

        //sf_color_list.push_back(sf::Color(rc,gc,bc));
        dev_color tempColor;
        tempColor.r=rc;
        tempColor.g=gc;
        tempColor.b=bc;
        host_color_list.push_back(tempColor);
    }

    for (int i=255; i<255+255; i++)
    {
        const tinycolormap::Color color = tinycolormap::GetColor((float(i-255))/half_number_colors, tinycolormap::ColormapType::Magma);
        float rc=color.r()*255;
        float gc=color.g()*255;
        float bc=color.b()*255;

        //sf_color_list.push_back(sf::Color(rc,gc,bc));
        dev_color tempColor;
        tempColor.r=rc;
        tempColor.g=gc;
        tempColor.b=bc;
        host_color_list.push_back(tempColor);
    }

    Init();


}



//--------------------------------------------------------------------------------//
//                        CUDA INITIALIZER FUNCTIONS                              //
//--------------------------------------------------------------------------------//

//provide LBM constants for d2q9 style nodes
//assumes positive is up and right, whereas our program assumes positive down and right.
void MainLoop::init_d2q9(d2q9_node* d2q9)
{
    d2q9[0].ex =  0;	d2q9[0].ey =  0;	d2q9[0].wt = 4.0 /  9.0;	d2q9[0].op = 0;
    d2q9[1].ex =  1;	d2q9[1].ey =  0;	d2q9[1].wt = 1.0 /  9.0;	d2q9[1].op = 3;
    d2q9[2].ex =  0;	d2q9[2].ey =  1;	d2q9[2].wt = 1.0 /  9.0;	d2q9[2].op = 4;
    d2q9[3].ex = -1;	d2q9[3].ey =  0;	d2q9[3].wt = 1.0 /  9.0;	d2q9[3].op = 1;
    d2q9[4].ex =  0;	d2q9[4].ey = -1;	d2q9[4].wt = 1.0 /  9.0;	d2q9[4].op = 2;
    d2q9[5].ex =  1;	d2q9[5].ey =  1;	d2q9[5].wt = 1.0 / 36.0;	d2q9[5].op = 7;
    d2q9[6].ex = -1;	d2q9[6].ey =  1;	d2q9[6].wt = 1.0 / 36.0;	d2q9[6].op = 8;
    d2q9[7].ex = -1;	d2q9[7].ey = -1;	d2q9[7].wt = 1.0 / 36.0;	d2q9[7].op = 5;
    d2q9[8].ex =  1;	d2q9[8].ey = -1;	d2q9[8].wt = 1.0 / 36.0;	d2q9[8].op = 6;
}


void MainLoop::zeroSite(lbm_node* array, int index)
{
    int dir = 0;
    for (dir = 0; dir < 9; dir += 1)
    {
        (array[index].f)[dir] = 0;
    }
    array[index].rho = 1;
    array[index].ux = 0;
    array[index].uy = 0;
}

void MainLoop::initBoundaries()
{
    int W = params.width;
    int H = params.height;
}



void MainLoop::initFluid() {
    int W = params.width;
    int H = params.height;

    //printf("velocity is %.6f my dude\n", v);

    barrier = (unsigned char*)calloc(W*H, sizeof(unsigned char));
    array1 = (lbm_node*)calloc(W*H, sizeof(lbm_node));
    array2 = (lbm_node*)calloc(W*H, sizeof(lbm_node));

    lbm_node* before = array1;

    d2q9_node* d2q9 = (d2q9_node*)calloc(9, sizeof(d2q9_node));
    init_d2q9(d2q9);

    DEBUG_PRINT(("\tTESTWEIGHT = %.6f", d2q9[dE].wt));

    int i;
    for (int x = 0; x < params.width; x++)
    {
        for (int y = 0; y < params.height; y++)
        {
            i = getIndex_cpu(x, y);

            before[i].rho = 1;
            before[i].ux = params.v;
            before[i].uy = 0;

            float v = params.v;


            int flow_height=32;
            int flow_width=2;
            if(y>params.height/2.0f-flow_height/2.0f and y<params.height/2.0f+flow_height/2.0f)
            {
                if(x<flow_width)
                {
                    before[i].rho = 1;
                    before[i].ux = params.v;
                    before[i].uy = params.v/2.0f;
                }

                else if(x>params.width-flow_width)
                {
                    before[i].rho = 1;
                    before[i].ux = -params.v;
                    v=-params.v;
                    before[i].uy = -params.v/2.0f;
                }
                else
                {
                    before[i].rho =1.0f;
                    before[i].ux = 0;
                    before[i].uy = 0;
                    v=0;
                }
            }
            else
            {
                before[i].rho =1.0f;
                before[i].ux = 0;
                before[i].uy = 0;
                v=0;
            }


            (before[i].f)[d0] = d2q9[d0].wt  * (1 - 1.5 * v * v);
            (before[i].f)[dE] = d2q9[dE].wt  * (1 + 3 * v + 3 * v * v);
            (before[i].f)[dW] = d2q9[dW].wt  * (1 - 3 * v + 3 * v * v);
            (before[i].f)[dN] = d2q9[dN].wt  * (1 - 1.5 * v * v);
            (before[i].f)[dS] = d2q9[dS].wt  * (1 - 1.5 * v * v);
            (before[i].f)[dNE] = d2q9[dNE].wt * (1 + 3 * v + 3 * v * v);
            (before[i].f)[dSE] = d2q9[dSE].wt * (1 + 3 * v + 3 * v * v);
            (before[i].f)[dNW] = d2q9[dNW].wt * (1 - 3 * v + 3 * v * v);
            (before[i].f)[dSW] = d2q9[dSW].wt * (1 - 3 * v + 3 * v * v);





            //-----------------------------
            /*
            int flow_height=15;
            int flow_width=45;
            if(y>params.height/2.0f-flow_height/2.0f and y<params.height/2.0f+flow_height/2.0f)
            {
                if(x<flow_width)
                {
                    before[i].rho = 1;
                    before[i].ux = params.v;
                    before[i].uy = 0;
                }

                else if(x>params.width-flow_width)
                {
                    before[i].rho = 1;
                    before[i].ux = -params.v;
                    before[i].uy = 0;
                }
                else
                {
                    before[i].rho =0;
                    before[i].ux = 0;
                    before[i].uy = 0;
                }
            }
            else
            {
                before[i].rho =0;
                before[i].ux = 0;
                before[i].uy = 0;
            }
            */

            //-----------------------------

            //before[i].rho = 1;
            //before[i].ux = params.v;
            //before[i].uy = 0;
        }
    }




    ierrSync = cudaMalloc(&d2q9_gpu, 9 * sizeof(d2q9_node));
    if (ierrSync != cudaSuccess) { DEBUG_PRINT(("Sync error: %s\n", cudaGetErrorString(ierrSync))); }
    ierrSync = cudaMalloc(&params_gpu, sizeof(parameter_set));
    if (ierrSync != cudaSuccess) { DEBUG_PRINT(("Sync error: %s\n", cudaGetErrorString(ierrSync))); }
    ierrSync = cudaMalloc(&barrier_gpu, sizeof(unsigned char)*W*H);
    if (ierrSync != cudaSuccess) { DEBUG_PRINT(("Sync error: %s\n", cudaGetErrorString(ierrSync))); }
    ierrSync = cudaMalloc(&array1_gpu, sizeof(lbm_node)*W*H);
    if (ierrSync != cudaSuccess) { DEBUG_PRINT(("Sync error: %s\n", cudaGetErrorString(ierrSync))); }
    ierrSync = cudaMalloc(&array2_gpu, sizeof(lbm_node)*W*H);
    if (ierrSync != cudaSuccess) { DEBUG_PRINT(("Sync error: %s\n", cudaGetErrorString(ierrSync))); }


    ierrSync = cudaMemcpy(d2q9_gpu, d2q9, sizeof(d2q9_node) * 9, cudaMemcpyHostToDevice);
    if (ierrSync != cudaSuccess) { DEBUG_PRINT(("Sync error: %s\n", cudaGetErrorString(ierrSync))); }
    ierrSync = cudaMemcpy(params_gpu, &params, sizeof(params), cudaMemcpyHostToDevice);
    if (ierrSync != cudaSuccess) { DEBUG_PRINT(("Sync error: %s\n", cudaGetErrorString(ierrSync))); }
    ierrSync = cudaMemcpy(barrier_gpu, barrier, sizeof(unsigned char)*W*H, cudaMemcpyHostToDevice);
    if (ierrSync != cudaSuccess) { DEBUG_PRINT(("Sync error: %s\n", cudaGetErrorString(ierrSync))); }
    ierrSync = cudaMemcpy(array1_gpu, array1, sizeof(lbm_node)*W*H, cudaMemcpyHostToDevice);
    if (ierrSync != cudaSuccess) { DEBUG_PRINT(("Sync error: %s\n", cudaGetErrorString(ierrSync))); }
    ierrSync = cudaMemcpy(array2_gpu, array2, sizeof(lbm_node)*W*H, cudaMemcpyHostToDevice);
    if (ierrSync != cudaSuccess) { DEBUG_PRINT(("Sync error: %s\n", cudaGetErrorString(ierrSync))); }


    N = display.tex_width*display.tex_height*4;
    ierrSync =(cudaMallocHost(&hostData, N * sizeof(float)));
    if (ierrSync != cudaSuccess) { DEBUG_PRINT(("Sync error: %s\n", cudaGetErrorString(ierrSync))); }
    ierrSync =(cudaMalloc(&devData, N * sizeof(float)));
    if (ierrSync != cudaSuccess) { DEBUG_PRINT(("Sync error: %s\n", cudaGetErrorString(ierrSync))); }

    ierrSync = cudaMalloc((void**)&dev_color_list, host_color_list.size()*sizeof(dev_color));
    ierrSync= cudaMemcpy(dev_color_list, host_color_list.data(), host_color_list.size()*sizeof(dev_color), cudaMemcpyHostToDevice);
    if (ierrSync != cudaSuccess) { DEBUG_PRINT(("Sync error: %s\n", cudaGetErrorString(ierrSync))); }

   // drawLineDiagonal();
   // drawSquare();
    loadBarrierFromFile();
    cudaMemcpy(barrier_gpu, barrier, sizeof(unsigned char)*params.width * params.height, cudaMemcpyHostToDevice);
    cudaMemcpy(params_gpu, &params, sizeof(params), cudaMemcpyHostToDevice);

    needsUpdate = 0;
    cudaDeviceSynchronize(); // Wait for the GPU to finish


    return;
}


void MainLoop::initFluid_flow() {
    int W = params.width;
    int H = params.height;

    array1 = (lbm_node*)calloc(W*H, sizeof(lbm_node));


    d2q9_node* d2q9 = (d2q9_node*)calloc(9, sizeof(d2q9_node));


    ierrSync = cudaMemcpy( d2q9, d2q9_gpu, sizeof(d2q9_node) * 9, cudaMemcpyDeviceToHost);
    if (ierrSync != cudaSuccess) { DEBUG_PRINT(("Sync error: %s\n", cudaGetErrorString(ierrSync))); }
    ierrSync = cudaMemcpy( array1, array1_gpu, sizeof(lbm_node)*W*H, cudaMemcpyDeviceToHost);
    if (ierrSync != cudaSuccess) { DEBUG_PRINT(("Sync error: %s\n", cudaGetErrorString(ierrSync))); }

     lbm_node* before = array1;


    int i;
    for (int x = 0; x < params.width; x++)
    {
        for (int y = 0; y < params.height; y++)
        {
            i = getIndex_cpu(x, y);

            before[i].rho = 1;
            before[i].ux = params.v;
            before[i].uy = 0;

            float v = params.v;


            int flow_height=32;
            int flow_width=5;

            if(y>params.height/2.0f-flow_height/2.0f and y<params.height/2.0f+flow_height/2.0f)
            {
                if(x<flow_width)
                {
                    before[i].rho = 1;
                    before[i].ux = params.v;
                    before[i].uy = params.v/2.0f;

                    (before[i].f)[d0] = d2q9[d0].wt  * (1 - 1.5 * v * v);
                    (before[i].f)[dE] = d2q9[dE].wt  * (1 + 3 * v + 3 * v * v);
                    (before[i].f)[dW] = d2q9[dW].wt  * (1 - 3 * v + 3 * v * v);
                    (before[i].f)[dN] = d2q9[dN].wt  * (1 - 1.5 * v * v);
                    (before[i].f)[dS] = d2q9[dS].wt  * (1 - 1.5 * v * v);
                    (before[i].f)[dNE] = d2q9[dNE].wt * (1 + 3 * v + 3 * v * v);
                    (before[i].f)[dSE] = d2q9[dSE].wt * (1 + 3 * v + 3 * v * v);
                    (before[i].f)[dNW] = d2q9[dNW].wt * (1 - 3 * v + 3 * v * v);
                    (before[i].f)[dSW] = d2q9[dSW].wt * (1 - 3 * v + 3 * v * v);
                }

                else if(x>params.width-flow_width)
                {
                    before[i].rho = 1;
                    before[i].ux = -params.v;
                    v=-params.v;
                    before[i].uy = -params.v/2.0f;

                    (before[i].f)[d0] = d2q9[d0].wt  * (1 - 1.5 * v * v);
                    (before[i].f)[dE] = d2q9[dE].wt  * (1 + 3 * v + 3 * v * v);
                    (before[i].f)[dW] = d2q9[dW].wt  * (1 - 3 * v + 3 * v * v);
                    (before[i].f)[dN] = d2q9[dN].wt  * (1 - 1.5 * v * v);
                    (before[i].f)[dS] = d2q9[dS].wt  * (1 - 1.5 * v * v);
                    (before[i].f)[dNE] = d2q9[dNE].wt * (1 + 3 * v + 3 * v * v);
                    (before[i].f)[dSE] = d2q9[dSE].wt * (1 + 3 * v + 3 * v * v);
                    (before[i].f)[dNW] = d2q9[dNW].wt * (1 - 3 * v + 3 * v * v);
                    (before[i].f)[dSW] = d2q9[dSW].wt * (1 - 3 * v + 3 * v * v);
                }





        }

       }

    }


    //ierrSync = cudaMemcpy(d2q9_gpu, d2q9, sizeof(d2q9_node) * 9, cudaMemcpyHostToDevice);
    //if (ierrSync != cudaSuccess) { DEBUG_PRINT(("Sync error: %s\n", cudaGetErrorString(ierrSync))); }
    ierrSync = cudaMemcpy(array1_gpu, array1, sizeof(lbm_node)*W*H, cudaMemcpyHostToDevice);
    if (ierrSync != cudaSuccess) { DEBUG_PRINT(("Sync error: %s\n", cudaGetErrorString(ierrSync))); }



    cudaDeviceSynchronize(); // Wait for the GPU to finish




}

//determine front and back lattice buffer orientation
//and launch all 3 LBM kernels
void MainLoop::kernelLauncher(unsigned char* image, bool& canDraw)
{
    if(params.need_update)
    {
        params.mode=display.current_Mode_Index;
        needsUpdate=1;
        params.need_update=false;
    }
    if (needsUpdate)
    {
        initFluid_flow();
        cudaMemcpy(barrier_gpu, barrier, sizeof(unsigned char)*params.width * params.height, cudaMemcpyHostToDevice);
        cudaMemcpy(params_gpu, &params, sizeof(params), cudaMemcpyHostToDevice);
        needsUpdate = 0;
        cudaDeviceSynchronize(); // Wait for the GPU to finish
    }

    bool resultKernel= step(params, params_gpu, d2q9_gpu,
                            array1_gpu,array2_gpu,barrier_gpu,
                            prex,  prey, image,
                            ierrAsync, ierrSync, canDraw, dev_color_list, display.colorZoom);


}

//-----------------------------------------------------------//
//                  BARRIER FUNCTIONS                        //
//-----------------------------------------------------------//

void MainLoop::clearBarriers()
{
    for (int i = 0;i < params.width;i++)
    {
        for (int j = 0; j < params.height;j++)
        {
            barrier[getIndex_cpu(i, j)] = 0;
        }
    }
}

void MainLoop::drawLineDiagonal()
{
    for (int i = 0; i < params.height/4; i++)
    {

        barrier[getIndex_cpu((params.width / 3) + (i / 3), params.height / 3 + i)] = 1;
    }
}

void MainLoop::drawSquare()
{
    for (int i = 0; i < params.height/16; i++)
    {
        for (int j = 0; j < params.height /16; j++)
        {
            //if(i==0 || i== params.height / 4-1 || j==0 || j == params.height / 4-1)
            barrier[getIndex_cpu(i+params.width/6, j+params.height * 2.5 / 8)] = 1;
        }
    }

    for (int i = 0; i < params.height/16; i++)
    {
        for (int j = 0; j < params.height / 16; j++)
        {
            //if(i==0 || i== params.height / 4-1 || j==0 || j == params.height / 4-1)
            barrier[getIndex_cpu(i+params.width/6, j+params.height * 3.5 / 8)] = 1;
        }
    }

    for (int i = 0; i < params.height/16; i++)
    {
        for (int j = 0; j < params.height / 16; j++)
        {
            //if(i==0 || i== params.height / 4-1 || j==0 || j == params.height / 4-1)
            barrier[getIndex_cpu(i+params.width/4, j+params.height * 3 / 8)] = 1;
        }
    }
}

void MainLoop::loadBarrierFromFile()
{
    sf::Image tempImage=display.SourceTexture.copyToImage();

    for (int i = 0; i < params.width; i++)
    {
        for (int j = 0; j < params.height; j++)
        {
           sf::Color tempColor=tempImage.getPixel(i,params.height-j);
           if( tempColor.r==255 and  tempColor.g==255 and  tempColor.b==255)
                barrier[getIndex_cpu(i, j)] = 0;
           else if( tempColor.r==0 and  tempColor.g==0 and  tempColor.b==255)
                barrier[getIndex_cpu(i, j)] = 2;
           else if( tempColor.r==0 and  tempColor.g==255 and  tempColor.b==0)
                barrier[getIndex_cpu(i, j)] = 3;
           else
                barrier[getIndex_cpu(i, j)] = 1;
        }
    }

}

void MainLoop::FreePointers()
{
    cudaFree(barrier_gpu);
    cudaFree(d2q9_gpu);
    cudaFree(params_gpu);
    cudaFree(array1_gpu);
    cudaFree(array2_gpu);
    cudaFree(devData);
    cudaFree(dev_color_list);

    free(array1);
    free(array2);
    //free(hostData);
    free(barrier);

}


bool MainLoop::Init(){

    //discover all Cuda-capable hardware
    int i = deviceQuery();
    //DEBUG_PRINT(("num devices is %d\n", i));

    if (i < 1)
    {
        //DEBUG_PRINT(("ERROR: no cuda capable hardware detected!\n"));
        getchar();
        return 0; //return if no cuda-capable hardware is present
    }

    //allocate memory and initialize fluid arrays
    getParams(&params); //change places
    initBoundaries();
    initFluid();


    return true;

}



void MainLoop::Run()
{
    sf::Clock deltaClock;

    bool openWindow=true;

    int tickCount=0;
    while (openWindow)
    {
        display.Clear_window();

        deltaClock.restart();

        sf::Clock clock;


        if(openWindow)
        {
            if(display.StartPausePhysics)
            {
                for (int i = 0; i < params.stepsPerRender; i++)
                {
                    bool canDraw=false;
                    if(i==params.stepsPerRender-1)
                        canDraw=true;
                    kernelLauncher(devData, canDraw);
                }
            }
            if(display.stepGraphPress)
            {
                for (int i = 0; i < params.stepsPerRender; i++)
                {
                    bool canDraw=false;
                    if(i==params.stepsPerRender-1)
                        canDraw=true;
                    kernelLauncher(devData, canDraw);
                }
                  display.stepGraphPress=false;
            }


             ierrSync = cudaMemcpy(hostData, devData, N*sizeof(float),cudaMemcpyDeviceToHost);
             if ( ierrSync  != cudaSuccess) { DEBUG_PRINT(("Sync error: %s\n", cudaGetErrorString(ierrSync))); }

             display.DrawLBMTex(hostData);

            display.solver_time = clock.getElapsedTime().asMicroseconds() * 0.001f;
            display.solver_fps= 1000.0f/display.solver_time;

            openWindow=display.ProcessEvents(params);
            display.Frame_draw();



        }

        tickCount++;

    }
    FreePointers();

}
