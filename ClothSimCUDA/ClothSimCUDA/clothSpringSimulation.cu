
#include "clothSpringSimulation.h"


__global__ void CalculateSpringsKernel(
	Vertex* vertPtr, 
	Spring* springPtr, 
	glm::vec3* posPtr, 
	glm::vec3* nrmPtr, 
	glm::vec4* colPtr, 
	const float* grav,
	const int N
	)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int v_cur = (i * gridDim.y * blockDim.y) + j;

	if (v_cur >= N)
		return;

	posPtr[springPtr[v_cur].idFirst].y = posPtr[springPtr[v_cur].idFirst].y - 0.00055f;
	//posPtr[springPtr[v_cur].idSecond].y -= 0.00055f;
}

__global__ void CalculateForcesKernel(
	Vertex* vertPtr, 
	Spring* springPtr, 
	glm::vec3* posPtr, 
	glm::vec3* nrmPtr, 
	glm::vec4* colPtr, 
	const float* grav, 
	const float delta,
	const unsigned int N
	)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int v_cur = (i * gridDim.y * blockDim.y) + j;

	if (v_cur >= N)
		return;
	
	vertPtr[v_cur].force = glm::vec3(0.0f, 0.0f, 0.0f);
	vertPtr[v_cur].velocity = glm::vec3(0.0f, 0.0f, 0.0f);
	float el = 0.1f;
	int id = vertPtr[v_cur].id;

	// calculate elasticity force for each neighbouring vertices
	for (int i = 0; i < VERTEX_NEIGHBOURING_VERTICES; ++i)
	{
		vertPtr[v_cur].force.x +=
			-el * (abs(posPtr[id].x - posPtr[vertPtr[v_cur].neighbours[i]].x) - vertPtr[v_cur].springLengths[i]) *
			(posPtr[id].x - posPtr[vertPtr[v_cur].neighbours[i]].x) /
			((abs(posPtr[id].x - posPtr[vertPtr[v_cur].neighbours[i]].x))) * 
			vertPtr[v_cur].neighbourMultipliers[i];

		vertPtr[v_cur].force.y +=
			-el * (abs(posPtr[id].y - posPtr[vertPtr[v_cur].neighbours[i]].y) - vertPtr[v_cur].springLengths[i]) *
			(posPtr[id].y - posPtr[vertPtr[v_cur].neighbours[i]].y) /
			((abs(posPtr[id].y - posPtr[vertPtr[v_cur].neighbours[i]].y))) *
			vertPtr[v_cur].neighbourMultipliers[i];

		vertPtr[v_cur].force.z +=
			-el * (abs(posPtr[id].z - posPtr[vertPtr[v_cur].neighbours[i]].z) - vertPtr[v_cur].springLengths[i]) *
			(posPtr[id].z - posPtr[vertPtr[v_cur].neighbours[i]].z) /
			((abs(posPtr[id].z - posPtr[vertPtr[v_cur].neighbours[i]].z))) *
			vertPtr[v_cur].neighbourMultipliers[i];
	}


	// calculate gravity force
	vertPtr[v_cur].force +=
		vertPtr[v_cur].mass * glm::vec3(0.0f, -(*grav) / 10.0f, 0.0f);

	// calculate air damp force
	vertPtr[v_cur].force +=
		-vertPtr[v_cur].dampCoeff * vertPtr[v_cur].velocity;

	// ?calculate repulsive force?

	// check hooks
	vertPtr[v_cur].force *= vertPtr[v_cur].lockMultiplier;


	// calculate acceleration and use Verelet integration to calculate position
	glm::vec3 newPos;
	glm::vec3 acc = vertPtr[v_cur].force / vertPtr[v_cur].mass;

	newPos = 2.0f * posPtr[id] - vertPtr[v_cur].prevPosition + acc * delta * delta;
	vertPtr[v_cur].prevPosition = posPtr[id];
	posPtr[id] = newPos;

	// update velocity
	vertPtr[v_cur].velocity = (newPos - vertPtr[v_cur].prevPosition) / delta;
}

__global__ void CalculatePositionsKernel(
	Vertex* vertPtr, 
	Spring* springPtr, 
	glm::vec3* posPtr, 
	glm::vec3* nrmPtr, 
	glm::vec4* colPtr, 
	const float* grav,
	const int N
	)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int v_cur = (i * gridDim.y * blockDim.y) + j;

	if (v_cur >= N)
		return;
}

__global__ void CalculateNormalsKernel(
	Vertex* vertPtr, 
	Spring* springPtr, 
	glm::vec3* posPtr, 
	glm::vec3* nrmPtr, 
	glm::vec4* colPtr, 
	const float* grav, 
	const int N
	)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int v_cur = (i * gridDim.y * blockDim.y) + j;

	if (v_cur >= N)
		return;
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
	unsigned int edgesLength,
	glm::vec3* vertexPositionPtr,
	glm::vec3* vertexNormalPtr,
	glm::vec4* vertexColorPtr
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
	m_springCount = (m_allEdgesLength - 1) * (m_allEdgesWidth) + (m_allEdgesWidth - 1) * m_allEdgesLength;
	m_posPtr = vertexPositionPtr;
	m_nrmPtr = vertexNormalPtr;
	m_colPtr = vertexColorPtr;

	// generate vertex and spring arrays, to help with computations

	m_vertices = new Vertex[m_vertexCount];

	for (int i = 0; i < m_vertexCount; ++i)
	{
		m_vertices[i].id = i;
		m_vertices[i].mass = VERTEX_MASS;
		m_vertices[i].lockMultiplier = 1.0f;
		m_vertices[i].prevPosition = m_posPtr[i];
		m_vertices[i].dampCoeff = VERTEX_DAMP;

		// calculating neighbouring vertices ids and spring lengths

		float lengthX = abs(m_posPtr[0].x - m_posPtr[m_vertexCount - 1].x) / (float)(m_allEdgesLength - 1);
		float lengthZ = abs(m_posPtr[0].z - m_posPtr[m_vertexCount - 1].z) / (float)(m_allEdgesWidth - 1);

		// upper
		m_vertices[i].neighbours[0] = (i - 1) % m_vertexCount;
		if (i % m_allEdgesLength)
		{
			m_vertices[i].neighbourMultipliers[0] = 1.0f;
			m_vertices[i].springLengths[0] = lengthZ;
		}	
		else
		{
			m_vertices[i].neighbourMultipliers[0] = 0.0f;
			m_vertices[i].springLengths[0] = 0.0f;
		}

		// lower
		m_vertices[i].neighbours[1] = (i + 1) % m_vertexCount;
		if (i % m_allEdgesLength != (m_allEdgesLength - 1))
		{
			m_vertices[i].neighbourMultipliers[1] = 1.0f;
			m_vertices[i].springLengths[1] = lengthZ;
		}
		else
		{
			m_vertices[i].neighbourMultipliers[1] = 0.0f;
			m_vertices[i].springLengths[1] = 0.0f;
		}

		// left
		m_vertices[i].neighbours[2] = (i - m_allEdgesLength) % m_vertexCount;
		if (i >= m_allEdgesLength)
		{
			m_vertices[i].neighbourMultipliers[2] = 1.0f;
			m_vertices[i].springLengths[1] = lengthX;
		}
		else
		{
			m_vertices[i].neighbourMultipliers[2] = 0.0f;
			m_vertices[i].springLengths[2] = 0.0f;
		}

		// right
		m_vertices[i].neighbours[3] = (i + m_allEdgesLength) % m_vertexCount;
		if (i < (m_vertexCount - m_allEdgesLength))
		{
			m_vertices[i].neighbourMultipliers[3] = 1.0f;
			m_vertices[i].springLengths[3] = lengthX;
		}
		else
		{
			m_vertices[i].neighbourMultipliers[3] = 0.0f;
			m_vertices[i].springLengths[3] = 0.0f;
		}
	}


	m_springs = new Spring[m_springCount];

	for (int i = 0, s = 0; i < m_vertexCount; ++i)
	{
		// do I create "lower" spring?
		if (i % m_allEdgesLength != (m_allEdgesLength - 1))
		{
			m_springs[s].idFirst = i;
			m_springs[s].idSecond = i + 1;

			m_springs[s].baseLength = glm::length(m_posPtr[i] - m_posPtr[i + 1]);
			m_springs[s].elasticity = SPRING_ELASTICITY;

			++s;
		}

		// do I create "right" spring?
		if (i < (m_vertexCount - m_allEdgesLength))
		{
			m_springs[s].idFirst = i;
			m_springs[s].idSecond = i + m_allEdgesLength;

			m_springs[s].baseLength = glm::length(m_posPtr[i] - m_posPtr[i + 1]);
			m_springs[s].elasticity = SPRING_ELASTICITY;

			++s;
		}
	}

	//////////////////////////////

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

	// Allocate GPU buffers for six vectors (2 input for helpers, 3 input for buffers) and one float (gravity)
	cudaStatus = cudaMalloc((void**)&i_vertexPtr, m_vertexCount * sizeof(Vertex));
	if (cudaStatus != cudaSuccess) {
		printf("CUDA: cudaMalloc for vertex helper buffer failed!");
		FreeMemory();
		return cudaStatus;
	}

	cudaStatus = cudaMalloc((void**)&i_springPtr, m_springCount * sizeof(Spring));
	if (cudaStatus != cudaSuccess) {
		printf("CUDA: cudaMalloc for spring helper buffer failed!");
		FreeMemory();
		return cudaStatus;
	}

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

	// copy helper buffers to device memory
	cudaStatus = cudaMemcpy(i_vertexPtr, m_vertices, m_vertexCount * sizeof(Vertex), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		FreeMemory();
		return cudaStatus;
	}

	cudaStatus = cudaMemcpy(i_springPtr, m_springs, m_springCount * sizeof(Spring), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		FreeMemory();
		return cudaStatus;
	}
}

unsigned int clothSpringSimulation::ClothSpringSimulationUpdate(float gravity, double delta)
{
	// Add vectors in parallel.
	cudaError_t cudaStatus = CalculateForces(gravity, delta);
	if (cudaStatus != cudaSuccess) {
		printf("CUDA: AddWithCuda failed!");
		return CS_ERR_CLOTHSIMULATOR_CUDA_FAILED;
	}

	//printf("%f\n",m_posPtr[0].y);
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


inline cudaError_t clothSpringSimulation::CalculateForces(float gravity, double delta)
{
	cudaError_t status;

	// copy vertex data and simulation variables to device memory

	status = cudaMemcpy(i_vertexPtr, m_vertices, m_vertexCount * sizeof(Vertex), cudaMemcpyHostToDevice);
	if (status != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		FreeMemory();
		return status;
	}

	/*
	status = cudaMemcpy(i_springPtr, m_springs, m_springCount * sizeof(Spring), cudaMemcpyHostToDevice);
	if (status != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		FreeMemory();
		return status;
	}
	*/

	status = cudaMemcpy(i_posPtr, m_posPtr, m_vertexCount * m_vertexPositionSize, cudaMemcpyHostToDevice);
	if (status != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		FreeMemory();
		return status;
	}

	status = cudaMemcpy(i_nrmPtr, m_nrmPtr, m_vertexCount * m_vertexNormalSize, cudaMemcpyHostToDevice);
	if (status != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		FreeMemory();
		return status;
	}

	status = cudaMemcpy(i_colPtr, m_colPtr, m_vertexCount * m_vertexColorSize, cudaMemcpyHostToDevice);
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
	int p = m_deviceProperties->warpSize;
	int sX = (m_allEdgesWidth - 1) * m_allEdgesLength;
	int sY = (m_allEdgesLength - 1) * m_allEdgesWidth;
	dim3 gridVerts((m_allEdgesWidth + p - 1) / p, (m_allEdgesLength + p - 1) / p, 1);
	dim3 gridSprings((sX + p - 1) / p, (sY + p - 1) / p, 1);
	dim3 blockVerts(p, p, 1);
	dim3 blockSprings(p, p, 1);

	//CalculateSpringsKernel << < gridSprings, blockSprings >> > (i_vertexPtr, i_springPtr, i_posPtr, i_nrmPtr, i_colPtr, i_gravPtr, m_springCount);
	CalculateForcesKernel << < gridVerts, blockVerts >> > (i_vertexPtr, i_springPtr, i_posPtr, i_nrmPtr, i_colPtr, i_gravPtr, delta / 1000.0f, m_vertexCount);
	//CalculatePositionsKernel << < gridVerts, blockVerts >> > (i_vertexPtr, i_springPtr, i_posPtr, i_nrmPtr, i_colPtr, i_gravPtr, m_vertexCount);
	//CalculateNormalsKernel << < gridVerts, blockVerts >> > (i_vertexPtr, i_springPtr, i_posPtr, i_nrmPtr, i_colPtr, i_gravPtr, m_vertexCount);

	// Check for any errors launching the kernel
	status = cudaGetLastError();
	if (status != cudaSuccess) {
		fprintf(stderr, "AddKernel launch failed: %s\n", cudaGetErrorString(status));
		FreeMemory();
		return status;
	}

	// copy calculated data out of device memory


	status = cudaMemcpy(m_vertices, i_vertexPtr, m_vertexCount * sizeof(Vertex), cudaMemcpyDeviceToHost);
	if (status != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		FreeMemory();
		return status;
	}

	/*
	status = cudaMemcpy(m_springs, i_springPtr, m_springCount * sizeof(Spring), cudaMemcpyDeviceToHost);
	if (status != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		FreeMemory();
		return status;
	}
	*/

	status = cudaMemcpy(m_posPtr, i_posPtr, m_vertexCount * m_vertexPositionSize, cudaMemcpyDeviceToHost);
	if (status != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		FreeMemory();
		return status;
	}

	status = cudaMemcpy(m_nrmPtr, i_nrmPtr, m_vertexCount * m_vertexNormalSize, cudaMemcpyDeviceToHost);
	if (status != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		FreeMemory();
		return status;
	}

	status = cudaMemcpy(m_colPtr, i_colPtr, m_vertexCount * m_vertexColorSize, cudaMemcpyDeviceToHost);
	if (status != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		FreeMemory();
		return status;
	}

	return status;
}


void clothSpringSimulation::FreeMemory()
{
	cudaFree(i_vertexPtr);
	cudaFree(i_springPtr);
	cudaFree(i_posPtr);
	cudaFree(i_nrmPtr);
	cudaFree(i_colPtr);
	cudaFree(i_gravPtr);
}