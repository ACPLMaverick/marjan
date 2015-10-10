
#include "clothSpringSimulation.h"


__global__ void AddKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

//////////////////////////////////////////////////////

clothSpringSimulation::clothSpringSimulation()
{
}

clothSpringSimulation::~clothSpringSimulation()
{

}

unsigned int clothSpringSimulation::ClothSpringSimulationInitialize(unsigned int vertexCount, unsigned int vertexPositionSize, unsigned int vertexNormalSize, unsigned int vertexColorSize)
{
	cudaError_t cudaStatus;

	// save data given to function
	m_vertexCount = vertexCount;
	m_vertexPositionSize = vertexPositionSize;
	m_vertexNormalSize = vertexNormalSize;
	m_vertexColorSize = vertexColorSize;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		printf("CUDA: cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		FreeMemory();
		return cudaStatus;
	}

	// Allocate GPU buffers for six vectors (3 input, 3 output) and one float (gravity)
	cudaStatus = cudaMalloc((void**)&dev_c, arraySize * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		printf("CUDA: cudaMalloc failed!");
		FreeMemory();
		return cudaStatus;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, arraySize * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		printf("CUDA: cudaMalloc failed!");
		FreeMemory();
		return cudaStatus;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, arraySize * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		printf("CUDA: cudaMalloc failed!");
		FreeMemory();
		return cudaStatus;
	}
}

unsigned int clothSpringSimulation::ClothSpringSimulationUpdate(glm::vec3* vertexPositionPtr, glm::vec3* vertexNormalPtr, glm::vec3* vertexColorPtr, float gravity)
{
	// Add vectors in parallel.
	cudaError_t cudaStatus = AddWithCuda(c, a, b, arraySize);
	if (cudaStatus != cudaSuccess) {
		printf("CUDA: AddWithCuda failed!");
		return CS_ERR_CLOTHSIMULATOR_CUDA_FAILED;
	}

	//printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
	//	c[0], c[1], c[2], c[3], c[4]);

	return CS_ERR_NONE;
}

unsigned int clothSpringSimulation::ClothSpringSimulationShutdown()
{
	cudaError_t cudaStatus;

	FreeMemory();

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		printf("CUDA: cudaDeviceReset failed!");
		return 1;
	}
}


/////////////////////////////////////////////////////


/////////////////////////////////////////////////////


// Helper function for using CUDA to add vectors in parallel.
inline cudaError_t clothSpringSimulation::AddWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
	cudaError_t cudaStatus;

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
		FreeMemory();
		return cudaStatus;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
		FreeMemory();
		return cudaStatus;
    }

    // Launch a kernel on the GPU with one thread for each element.
    AddKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "AddKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		FreeMemory();
		return cudaStatus;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching AddKernel!\n", cudaStatus);
		FreeMemory();
		return cudaStatus;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
		FreeMemory();
		return cudaStatus;
    }
    
	//FreeMemory();
	return cudaStatus;
}

void clothSpringSimulation::FreeMemory()
{
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);
}