#include "gstCamera.h"
#include "v4l2Camera.h"

#include "glDisplay.h"
#include "glTexture.h"

#include <stdio.h>
#include <signal.h>
#include <unistd.h>

#include "cudaNormalize.h"
#include "cudaFont.h"
#include "imageNet.h"

bool signal_recieved = false;

void sig_handler(int signo)
{
	if( signo == SIGINT )
	{
		printf("received SIGINT\n");
		signal_recieved = true;
	}
}


int main( int argc, char** argv )
{
    printf("trails\n  args (%i):  ", argc);

    for (int i = 0; i < argc; i++)
        printf("%i [%s]  ", i, argv[i]);

    printf("\n\n");

    // /*
    //  * parse network type from CLI arguments
    //  */
    // imageNet::NetworkType networkType = imageNet::GOOGLENET;

    // if( argc > 1 && strcmp(argv[1], "alexnet") == 0 )
    // 	networkType = imageNet::ALEXNET;

    if (signal(SIGINT, sig_handler) == SIG_ERR)
        printf("\ncan't catch SIGINT\n");

    //gstCamera* camera = gstCamera::Create();
    v4l2Camera* camera = v4l2Camera::Create("/dev/video1");

    if( !camera )
    {
    	printf("\ntrails:  failed to initialize video device\n");
    	return 1;
    }

    printf("\nntrails:  successfully initialized video device\n");
    printf("    width:  %u\n", camera->GetWidth());
    printf("   height:  %u\n", camera->GetHeight());
    printf("    depth:  %u (bpp)\n\n", camera->GetPixelDepth());

    // /*
    //  * create imageNet
    //  */
    // imageNet* net = imageNet::Create(networkType);

    // if( !net )
    // {
    // 	printf("imagenet-console:   failed to initialize imageNet\n");
    // 	return 0;
    // }

    glDisplay *display = glDisplay::Create();
    glTexture *texture = NULL;

    if (!display)
    {
        printf("\ntrails:  failed to create openGL display\n");
    }
    else
    {
        texture = glTexture::Create(camera->GetWidth(), camera->GetHeight(), GL_RGBA32F_ARB /*GL_RGBA8*/);

        if (!texture)
            printf("trails:  failed to create openGL texture\n");
    }

    cudaFont *font = cudaFont::Create();

    if (!camera->Open())
    {
        printf("\ntrails:  failed to open camera for streaming\n");
        return 0;
    }

    printf("\ntrails:  camera open for streaming\n");

    float confidence = 0.0f;

    while (!signal_recieved)
    {
        void *imgCPU = NULL;
        void *imgCUDA = NULL;
        void *imgRGBA = NULL;

        // get the latest frame
        imgRGBA = camera->Capture(1000);
        if (imgRGBA == nullptr)
            printf("\ntrails:  failed to capture frame\n");
        //else
        //	printf("trails:  recieved new frame  CPU=0x%p  GPU=0x%p\n", imgCPU, imgCUDA);

    //     // get the latest frame
    //     if (!camera->Capture(&imgCPU, &imgCUDA, 1000))
    //         printf("\ntrails:  failed to capture frame\n");
    //     //else
    //     //	printf("trails:  recieved new frame  CPU=0x%p  GPU=0x%p\n", imgCPU, imgCUDA);

    //     // convert from YUV to RGBA
    //     void *imgRGBA = NULL;

    //     if (!camera->ConvertRGBA(imgCUDA, &imgRGBA))
    //         printf("trails:  failed to convert from NV12 to RGBA\n");

    //     // classify image
    //     //const int img_class = net->Classify((float *)imgRGBA, camera->GetWidth(), camera->GetHeight(), &confidence);
        const int img_class = 10;

        if (img_class >= 0)
        {
            // printf("trails:  %2.5f%% class #%i (%s)\n", confidence * 100.0f, img_class, net->GetClassDesc(img_class));

            if (font != NULL)
            {
                char str[256];
                //sprintf(str, "%05.2f%% %s", confidence * 100.0f, net->GetClassDesc(img_class));
                sprintf(str, "%05.2f%% Test", confidence * 100.0f);

                font->RenderOverlay((float4*)imgRGBA, (float4*)imgRGBA, camera->GetWidth(), camera->GetHeight(),
                                    str, 0, 0, make_float4(255.0f, 255.0f, 255.0f, 255.0f));
            }

            if (display != NULL)
            {
                char str[256];
                //sprintf(str, "GIE build %x | %s | %s | %04.1f FPS", NV_GIE_VERSION, net->GetNetworkName(), net->HasFP16() ? "FP16" : "FP32", display->GetFPS());
                //sprintf(str, "GIE build %x | %s | %04.1f FPS | %05.2f%% %s", NV_GIE_VERSION, net->GetNetworkName(), display->GetFPS(), confidence * 100.0f, net->GetClassDesc(img_class));
                sprintf(str, "GIE build %x | %s | %s | %04.1f FPS", NV_GIE_VERSION, "Trails", "FP32", display->GetFPS());
                display->SetTitle(str);
            }
        }

        // update display
        if (display != NULL)
        {
            display->UserEvents();
            display->BeginRender();

            if (texture != NULL)
            {
                // rescale image pixel intensities for display
                CUDA(cudaNormalizeRGBA((float4 *)imgRGBA, make_float2(0.0f, 255.0f),
                                       (float4 *)imgRGBA, make_float2(0.0f, 1.0f),
                                       camera->GetWidth(), camera->GetHeight()));

                // map from CUDA to openGL using GL interop
                void *tex_map = texture->MapCUDA();

                if (tex_map != NULL)
                {
                    cudaMemcpy(tex_map, imgRGBA, texture->GetSize(), cudaMemcpyDeviceToDevice);
                    texture->Unmap();
                }

                // draw the texture
                texture->Render(100, 100);
            }

            display->EndRender();
        }
    }

    printf("\ntrails:  un-initializing video device\n");

    if (camera != NULL)
    {
        delete camera;
        camera = NULL;
    }

    if (display != NULL)
    {
        delete display;
        display = NULL;
    }

    printf("trails:  video device has been un-initialized.\n");
    printf("trails:  this concludes the test of the video device.\n");
    return 0;
}

