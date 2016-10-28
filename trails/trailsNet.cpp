#include "trailsNet.h"
#include "cudaMappedMemory.h"
#include "cudaResize.h"

trailsNet::trailsNet() : tensorNet()
{
	mOutputClasses = 0;
    mClassDesc = {"Left","Center","Right"};
}

trailsNet* trailsNet::Create(const char* prototxtPath, const char* modelPath)
{
    auto net = new trailsNet();

    if (!net->LoadNetwork(prototxtPath, modelPath, NULL, "data", "softmax"))
    {
		printf("failed to load %s\n", modelPath);
        return nullptr;
    }

	net->mOutputClasses = net->mOutputs[0].dims.c;

    return net;
}

// from imageNet.cu
cudaError_t cudaPreImageNet( float4* input, size_t inputWidth, size_t inputHeight, float* output, size_t outputWidth, size_t outputHeight, const float3& mean_value );

int trailsNet::Classify(float *rgba, uint32_t width, uint32_t height, float *confidence)
{
    if (CUDA_FAILED(cudaPreImageNet((float4 *)rgba, width, height, mInputCUDA, mWidth, mHeight,
                                    make_float3(104.0069879317889f, 116.66876761696767f, 122.6789143406786f))))
    {
        printf("imageNet::Classify() -- cudaPreImageNet failed\n");
        return -1;
    }

    // process with GIE
    void *inferenceBuffers[] = {mInputCUDA, mOutputs[0].CUDA};

    mContext->execute(1, inferenceBuffers);

    // determine the maximum class
    int classIndex = -1;
    float classMax = -1.0f;

    for (size_t n = 0; n < mOutputClasses; n++)
    {
        const float value = mOutputs[0].CPU[n];

        // if (value >= 0.01f)
        //     printf("class %04zu - %f  (%s)\n", n, value, mClassDesc[n].c_str());

        if (value > classMax)
        {
            classIndex = n;
            classMax = value;
        }
    }

    if (confidence != NULL)
        *confidence = classMax;

    //printf("\nmaximum class:  #%i  (%f) (%s)\n", classIndex, classMax, mClassDesc[classIndex].c_str());
    return classIndex;
}