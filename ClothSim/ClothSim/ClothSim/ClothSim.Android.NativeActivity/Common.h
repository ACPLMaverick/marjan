#pragma once
/*
Header file for common includes and switches
*/

#include <stdio.h>
#include <string>
#include <glm\glm\glm.hpp>

// MAKE SURE THERE'S ONLY AND EXACTLY ONE SWITCH TURNED ON AT A TIME
//#define BUILD_CUDA
#define BUILD_OPENCL

// MAKE SURE THERE'S ONLY AND EXACTLY ONE SWITCH TURNED ON AT A TIME
//#define BUILD_OPENGL
#define BUILD_OPENGLES


// error defines
#define CS_ERR_NONE 0
#define CS_ERR_UNKNOWN 1
#define CS_ERR_EGL_INITIALIZE_FAILED 2
#define CS_ERR_WINDOW_FAILED 3
#define CS_ANDROID_ERROR 4
#define CS_ERR_RESOURCE_READ_ERROR 5
#define CS_ERR_CLOTHCOLLIDER_MESH_OBTAINING_ERROR 6
#define CS_ERR_CLOTHSIMULATOR_MESH_OBTAINING_ERROR 7
#define CS_ERR_CLOTHSIMULATOR_COLLIDER_OBTAINING_ERROR 8
#define CS_ERR_CLOTHSIMULATOR_CUDA_FAILED 9
#define CS_ERR_SHUTDOWN_PENDING 10
//////////////////

// other helpful defines
#define LOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO, "AndroidProject1.NativeActivity", __VA_ARGS__))
#define LOGW(...) ((void)__android_log_print(ANDROID_LOG_WARN, "AndroidProject1.NativeActivity", __VA_ARGS__))

// structure definitions
struct TextureID
{
	int id;
	std::string name;
};

struct ShaderID
{
	int id;
	std::string name;

	int id_worldViewProj;
	int id_world;
	int id_worldInvTrans;
	int id_eyeVector;
	int id_lightDir;
	int id_lightDiff;
	int id_lightSpec;
	int id_lightAmb;
	int id_gloss;
	int id_highlight;
};
/////////////////

// functions

void Vec3Min(const glm::vec3* vec1, const glm::vec3* vec2, glm::vec3* ret);
void Vec3Max(const glm::vec3* vec1, const glm::vec3* vec2, glm::vec3* ret);
float Vec3LengthSquared(const glm::vec3*);

/////////////////