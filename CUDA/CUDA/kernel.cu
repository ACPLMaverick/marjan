
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <conio.h>

#include "Bitmap.h"

#define KERNEL_ARGS2(grid, block) <<< grid, block >>>
#define KERNEL_ARGS3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_ARGS4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>

cudaError_t BoxBlur(uchar3* dataPtr, unsigned int width, unsigned int height, unsigned int level);

__global__ void BlurKernel(uchar3* inputData, uchar3* outputData, unsigned int width, unsigned int height, unsigned int level)
{
    unsigned int offset = blockIdx.x*blockDim.x + threadIdx.x;

	int x = offset % width;
	int y = (offset - x) / width;
	unsigned int size = width*height;
	int iLevel = (int)level;

		float oR = 0.0f, oG = 0.0f, oB = 0.0f;
		unsigned int sum = 0;

		for (int i = -iLevel; i < iLevel + 1; ++i)
		{
			for (int j = -iLevel; j < iLevel + 1; ++j)
			{
				if ((x + i) >= 0 &&
					(x + i) < width &&
					(y + j) >= 0 &&
					(y + j) < height)
				{
					const int currentOffset = (offset + i + j * width);
					oR += ((float)inputData[currentOffset].z / 255.0f);
					oG += ((float)inputData[currentOffset].y / 255.0f);
					oB += ((float)inputData[currentOffset].x / 255.0f);
					++sum;
				}
				
			}
		}
		outputData[offset].z = (unsigned char)(oR / sum * 255.0f);
		outputData[offset].y = (unsigned char)(oG / sum * 255.0f);
		outputData[offset].x = (unsigned char)(oB / sum * 255.0f);
}

int main()
{
	unsigned int blurLevel;
	char path[64];
	char level[4];
	printf("HIGH PERFORMANCE CUDA-POWERED ULTRA BOX BLUR. \nGive file path: ");
	gets(path);
	printf("Give blur amount: ");
	gets(level);
	blurLevel = strtoul(level, nullptr, 0);

	Bitmap bitmap;
	bool bitmapResult;
	bitmapResult = bitmap.Load("canteen.bmp");
	if (!bitmapResult)
	{
		fprintf(stderr, "Bitmap loading failed!");
		return 1;
	}


    // Add vectors in parallel.
	cudaError_t cudaStatus = BoxBlur(bitmap.GetPtr(), bitmap.GetWidth(), bitmap.GetHeight(), blurLevel);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "BoxBlur failed!");
		getch();
        return 1;
    }

	printf("BoxBlur succeeded!");

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
		getch();
        return 1;
    }


	bitmapResult = bitmap.Save("canteen_b.bmp");
	if (!bitmapResult)
	{
		fprintf(stderr, "Bitmap saving failed!");
		getch();
		return 1;
	}

	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t BoxBlur(uchar3* dataPtr, unsigned int width, unsigned int height, unsigned int level)
{
	uchar3* input;
	uchar3* output;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&input, width * height * sizeof(uchar3));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	cudaStatus = cudaMemcpy(input, dataPtr, width * height* sizeof(uchar3), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	cudaStatus = cudaMalloc((void**)&output, width * height* sizeof(uchar3));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	dim3 blockDims(1024, 1, 1);
	dim3 gridDims((unsigned int)ceil((double)(width * height / blockDims.x)), 1, 1);
	//unsigned int nThreads = 64;
    // Launch a kernel on the GPU with one thread for each element.
    BlurKernel KERNEL_ARGS2(blockDims, gridDims) (input, output, width, height, level);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(dataPtr, output, width * height* sizeof(uchar3), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
	cudaFree(input);
	cudaFree(output);
    
    return cudaStatus;
}
