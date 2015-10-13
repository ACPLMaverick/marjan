
#include "clothSpringSimulation.h"


__global__ void AddKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

__global__ void CalculateForcesKernel(glm::vec3* posPtr, glm::vec3* nrmPtr, glm::vec4* colPtr, const float* grav)
{
	int i = blockIdx.x;
	int j = blockIdx.y;

	int v_cur = (i * gridDim.y) + j;
	
	// array of neighbouring four vertices, i.e. their positions in the grid
	// -1 in the array means that vertex is nonexistant
	// this array goes as follows: UP, DOWN, LEFT, RIGHT
	int v[4];
	v[0] = imaxi(-1, v_cur - 1);

	// test
	posPtr[v_cur].y -= 0.00025f;
	colPtr[v_cur].r = fmaxf(0.0f, colPtr[v_cur].r - 0.0001f);
	///
}

//////////////////////////////////////////////////////

clothSpringSimulation::clothSpringSimulation()
{
}

clothSpringSimulation::~clothSpringSimulation()
{

}

unsigned int clothSpringSimulation::ClothSpringSimulationInitialize(
	unsigned int vertexPositionSize,
	unsigned int vertexNormalSize,
	unsigned int vertexColorSize,
	unsigned int edgesWidth,
	unsigned int edgesLength
	)
{
	cudaError_t cudaStatus;

	// save data given to function
	m_vertexPositionSize = vertexPositionSize;
	m_vertexNormalSize = vertexNormalSize;
	m_vertexColorSize = vertexColorSize;
	m_allEdgesWidth = edgesWidth;
	m_allEdgesLength = edgesLength;
	m_vertexCount = m_allEdgesLength * m_allEdgesWidth;

	// Get Device info
	m_deviceProperties = new cudaDeviceProp;
	cudaStatus = cudaGetDeviceProperties(m_deviceProperties, 0);
	if (cudaStatus != cudaSuccess) {
		printf("CUDA: cudaGetDeviceProperties failed!  Do you have a CUDA-capable GPU installed?");
		FreeMemory();
		return cudaStatus;
	}

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		printf("CUDA: cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		FreeMemory();
		return cudaStatus;
	}

	// Allocate GPU buffers for six vectors (3 input, /*3 output*/) and one float (gravity)
	cudaStatus = cudaMalloc((void**)&i_posPtr, m_vertexCount * m_vertexPositionSize);
	if (cudaStatus != cudaSuccess) {
		printf("CUDA: cudaMalloc for position buffer failed!");
		FreeMemory();
		return cudaStatus;
	}

	cudaStatus = cudaMalloc((void**)&i_nrmPtr, m_vertexCount * m_vertexNormalSize);
	if (cudaStatus != cudaSuccess) {
		printf("CUDA: cudaMalloc for normal buffer failed!");
		FreeMemory();
		return cudaStatus;
	}

	cudaStatus = cudaMalloc((void**)&i_colPtr, m_vertexCount * m_vertexColorSize);
	if (cudaStatus != cudaSuccess) {
		printf("CUDA: cudaMalloc for color buffer failed!");
		FreeMemory();
		return cudaStatus;
	}

	cudaStatus = cudaMalloc((void**)&i_gravPtr, sizeof(float));
	if (cudaStatus != cudaSuccess) {
		printf("CUDA: cudaMalloc for gravity variable failed!");
		FreeMemory();
		return cudaStatus;
	}
}

unsigned int clothSpringSimulation::ClothSpringSimulationUpdate(glm::vec3* vertexPositionPtr, glm::vec3* vertexNormalPtr, glm::vec4* vertexColorPtr, float gravity)
{
	// Add vectors in parallel.
	cudaError_t cudaStatus = CalculateForces(vertexPositionPtr, vertexNormalPtr, vertexColorPtr, gravity);
	if (cudaStatus != cudaSuccess) {
		printf("CUDA: AddWithCuda failed!");
		return CS_ERR_CLOTHSIMULATOR_CUDA_FAILED;
	}

	//printf("%f\n",vertexPositionPtr[0].y);
	//printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
	//	c[0], c[1], c[2], c[3], c[4]);

	return CS_ERR_NONE;
}

unsigned int clothSpringSimulation::ClothSpringSimulationShutdown()
{
	cudaError_t cudaStatus;

	delete m_deviceProperties;

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

inline cudaError_t clothSpringSimulation::CalculateForces(glm::vec3* vertexPositionPtr, glm::vec3* vertexNormalPtr, glm::vec4* vertexColorPtr, float gravity)
{
	cudaError_t status;

	// copy vertex data and simulation variables to device memory

	status = cudaMemcpy(i_posPtr, vertexPositionPtr, m_vertexCount * m_vertexPositionSize, cudaMemcpyHostToDevice);
	if (status != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		FreeMemory();
		return status;
	}

	status = cudaMemcpy(i_nrmPtr, vertexNormalPtr, m_vertexCount * m_vertexNormalSize, cudaMemcpyHostToDevice);
	if (status != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		FreeMemory();
		return status;
	}

	status = cudaMemcpy(i_colPtr, vertexColorPtr, m_vertexCount * m_vertexColorSize, cudaMemcpyHostToDevice);
	if (status != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		FreeMemory();
		return status;
	}

	status = cudaMemcpy(i_gravPtr, &gravity, sizeof(float), cudaMemcpyHostToDevice);
	if (status != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		FreeMemory();
		return status;
	}

	// launch kernel
	dim3 grid(m_allEdgesWidth, m_allEdgesLength, 1);
	dim3 block(1, 1, 1);
	CalculateForcesKernel <<< grid, block >>> (i_posPtr, i_nrmPtr, i_colPtr, i_gravPtr);

	// Check for any errors launching the kernel
	status = cudaGetLastError();
	if (status != cudaSuccess) {
		fprintf(stderr, "AddKernel launch failed: %s\n", cudaGetErrorString(status));
		FreeMemory();
		return status;
	}

	// copy calculated data out of device memory
	status = cudaMemcpy(vertexPositionPtr, i_posPtr, m_vertexCount * m_vertexPositionSize, cudaMemcpyDeviceToHost);
	if (status != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		FreeMemory();
		return status;
	}

	status = cudaMemcpy(vertexNormalPtr, i_nrmPtr, m_vertexCount * m_vertexNormalSize, cudaMemcpyDeviceToHost);
	if (status != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		FreeMemory();
		return status;
	}

	status = cudaMemcpy(vertexColorPtr, i_colPtr, m_vertexCount * m_vertexColorSize, cudaMemcpyDeviceToHost);
	if (status != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		FreeMemory();
		return status;
	}

	return status;
}

// Helper function for using CUDA to add vectors in parallel.
/*
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
*/

void clothSpringSimulation::FreeMemory()
{
	cudaFree(i_posPtr);
	cudaFree(i_nrmPtr);
	cudaFree(i_colPtr);
	cudaFree(i_gravPtr);
}