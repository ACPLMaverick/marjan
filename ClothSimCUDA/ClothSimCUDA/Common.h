#pragma once
/*
Header file for common includes and switches
*/

#include <stdio.h>
#include <string>

// MAKE SURE THERE'S ONLY AND EXACTLY ONE SWITCH TURNED ON AT A TIME
#define BUILD_CUDA
//#define BUILD_OPENCL

// MAKE SURE THERE'S ONLY AND EXACTLY ONE SWITCH TURNED ON AT A TIME
#define BUILD_OPENGL
//#define BUILD_OPENGLES


// error defines
#define CS_ERR_NONE 0
#define CS_ERR_UNKNOWN 1
#define CS_ERR_GLFW_INITIALIZE_FAILED 2
#define CS_ERR_WINDOW_FAILED 3
#define CS_ERR_GLEW_INITIALIZE_FAILED 4
#define CS_ERR_RESOURCE_READ_ERROR 5
//////////////////

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