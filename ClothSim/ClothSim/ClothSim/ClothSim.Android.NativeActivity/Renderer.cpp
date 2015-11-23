#include "Renderer.h"

Renderer::Renderer()
{
}

Renderer::Renderer(const Renderer*)
{
}

Renderer::~Renderer()
{
}


unsigned int Renderer::Initialize()
{
	unsigned int err = CS_ERR_NONE;

	// initialize OpenGL ES and EGL

	/*
	* Here specify the attributes of the desired configuration.
	* Below, we select an EGLConfig with at least 8 bits per color
	* component compatible with on-screen windows
	*/
	const EGLint attribs[] = {
		EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
		EGL_BLUE_SIZE, 8,
		EGL_GREEN_SIZE, 8,
		EGL_RED_SIZE, 8,
		EGL_NONE
	};
	// dont forget about vsync here
	EGLint w, h, format;
	EGLint numConfigs;
	EGLConfig config;
	EGLSurface surface;
	EGLContext context;

	Engine* engine = System::GetInstance()->GetEngineData();

	EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);

	eglInitialize(display, 0, 0);

	/* Here, the application chooses the configuration it desires. In this
	* sample, we have a very simplified selection process, where we pick
	* the first EGLConfig that matches our criteria */
	eglChooseConfig(display, attribs, &config, 1, &numConfigs);

	/* EGL_NATIVE_VISUAL_ID is an attribute of the EGLConfig that is
	* guaranteed to be accepted by ANativeWindow_setBuffersGeometry().
	* As soon as we picked a EGLConfig, we can safely reconfigure the
	* ANativeWindow buffers to match, using EGL_NATIVE_VISUAL_ID. */
	eglGetConfigAttrib(display, config, EGL_NATIVE_VISUAL_ID, &format);

	ANativeWindow_setBuffersGeometry(engine->app->window, 0, 0, format);

	surface = eglCreateWindowSurface(display, config, engine->app->window, NULL);
	context = eglCreateContext(display, config, NULL, NULL);

	if (eglMakeCurrent(display, surface, surface, context) == EGL_FALSE) {
		LOGW("Unable to eglMakeCurrent");
		return -1;
	}

	eglQuerySurface(display, surface, EGL_WIDTH, &w);
	eglQuerySurface(display, surface, EGL_HEIGHT, &h);

	engine->display = display;
	engine->context = context;
	engine->surface = surface;
	engine->width = w;
	engine->height = h;

	// Shaders Loading
	
	ResourceManager::GetInstance()->LoadShader(&SN_BASIC);
	ResourceManager::GetInstance()->LoadShader(&SN_BASIC, &SN_WIREFRAME);
	ResourceManager::GetInstance()->LoadShader(&SN_FONT);
	m_shaderID = ResourceManager::GetInstance()->GetShader((unsigned int)0);

	m_mode = BASIC;

	//////////// options go here

	glClearColor(CSSET_CLEAR_COLORS[0], CSSET_CLEAR_COLORS[1], CSSET_CLEAR_COLORS[2], CSSET_CLEAR_COLORS[3]);
	glEnable(GL_DEPTH_TEST);

	if (CSSET_BACKFACE_CULLING)
	{
		glEnable(GL_CULL_FACE);
		glCullFace(GL_BACK);
		glFrontFace(GL_CCW);
	}

	glEnable(GL_DITHER);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_FASTEST);

	glEnable(GL_STENCIL);
	glStencilFunc(GL_LEQUAL, 0, 0xFF);
	glStencilOp(GL_REPLACE, GL_KEEP, GL_KEEP);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	
	glDepthFunc(GL_LEQUAL);

	/////////////////////////

	return err;
}

unsigned int Renderer::Shutdown()
{
	unsigned int err = CS_ERR_NONE;
	Engine* engine = System::GetInstance()->GetEngineData();

	if (engine->display != EGL_NO_DISPLAY) {
		eglMakeCurrent(engine->display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
		if (engine->context != EGL_NO_CONTEXT) {
			eglDestroyContext(engine->display, engine->context);
		}
		if (engine->surface != EGL_NO_SURFACE) {
			eglDestroySurface(engine->display, engine->surface);
		}
		eglTerminate(engine->display);
	}
	engine->animating = 0;
	engine->display = EGL_NO_DISPLAY;
	engine->context = EGL_NO_CONTEXT;
	engine->surface = EGL_NO_SURFACE;

	return err;
}

unsigned int Renderer::Run()
{
	unsigned int err = CS_ERR_NONE;
	Engine* engine = System::GetInstance()->GetEngineData();

	if (engine->display == NULL) 
	{
		// No display.
		return CS_ERR_UNKNOWN;
	}

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Drawing models with basic shader
	/*
	if (m_mode == BASIC || m_mode == BASIC_WIREFRAME)
	{*/
		m_shaderID = ResourceManager::GetInstance()->GetShader(&SN_BASIC);
		glUseProgram(m_shaderID->id);

		//glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

		System::GetInstance()->GetCurrentScene()->Draw();
		/*
	}
	
	// Drawing wireframe
	if (m_mode == WIREFRAME || m_mode == BASIC_WIREFRAME)
	{
		m_shaderID = ResourceManager::GetInstance()->GetShader(&SN_WIREFRAME);
		glUseProgram(m_shaderID->id);

		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

		System::GetInstance()->GetCurrentScene()->Draw();

		glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);

		System::GetInstance()->GetCurrentScene()->Draw();
	}
	*/

	eglSwapBuffers(engine->display, engine->surface);

	return err;
}

void Renderer::SetDrawMode(DrawMode mode)
{
	m_mode = mode;
}



void Renderer::SetCurrentShader(ShaderID* id)
{
	m_shaderID = id;
	glUseProgram(id->id);
}



ShaderID* Renderer::GetCurrentShaderID()
{
	return m_shaderID;
}


DrawMode Renderer::GetDrawMode()
{
	return m_mode;
}



void Renderer::LoadShaders(const string* vertexFilePath, const string* fragmentFilePath, const string* newName, ShaderID* n)
{
	string extension = ".glsl";

	GLuint vertexShaderID = glCreateShader(GL_VERTEX_SHADER);
	GLuint fragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);

	// Read vs code from file
	string vertexShaderCode;
	ifstream vertexShaderStream((*vertexFilePath + extension).c_str(), ios::in);
	if (vertexShaderStream.is_open())
	{
		string Line = "";
		while (getline(vertexShaderStream, Line))
			vertexShaderCode += "\n" + Line;
		vertexShaderStream.close();
	}

	string fragmentShaderCode;
	ifstream fragmentShaderStream((*fragmentFilePath + extension).c_str(), ios::in);
	if (fragmentShaderStream.is_open())
	{
		string Line = "";
		while (getline(fragmentShaderStream, Line))
			fragmentShaderCode += "\n" + Line;
		fragmentShaderStream.close();
	}

	GLint result = GL_FALSE;
	int infoLogLength;

	// Compile Vertex Shader
	LOGI("Compiling shader : %s\n", vertexFilePath->c_str());
	char const * VertexSourcePointer = vertexShaderCode.c_str();
	glShaderSource(vertexShaderID, 1, &VertexSourcePointer, NULL);
	glCompileShader(vertexShaderID);

	// Check Vertex Shader
	glGetShaderiv(vertexShaderID, GL_COMPILE_STATUS, &result);
	glGetShaderiv(vertexShaderID, GL_INFO_LOG_LENGTH, &infoLogLength);
	std::vector<char> VertexShaderErrorMessage(infoLogLength);
	glGetShaderInfoLog(vertexShaderID, infoLogLength, NULL, &VertexShaderErrorMessage[0]);
	LOGW("%s\n", &VertexShaderErrorMessage[0]);

	// Compile Fragment Shader
	LOGI("Compiling shader : %s\n", fragmentFilePath->c_str());
	char const * FragmentSourcePointer = fragmentShaderCode.c_str();
	glShaderSource(fragmentShaderID, 1, &FragmentSourcePointer, NULL);
	glCompileShader(fragmentShaderID);

	// Check Fragment Shader
	glGetShaderiv(fragmentShaderID, GL_COMPILE_STATUS, &result);
	glGetShaderiv(fragmentShaderID, GL_INFO_LOG_LENGTH, &infoLogLength);
	std::vector<char> FragmentShaderErrorMessage(infoLogLength);
	glGetShaderInfoLog(fragmentShaderID, infoLogLength, NULL, &FragmentShaderErrorMessage[0]);
	LOGW("%s\n", &FragmentShaderErrorMessage[0]);

	// Link the program
	LOGI("Linking program\n");
	GLuint ProgramID = glCreateProgram();
	glAttachShader(ProgramID, vertexShaderID);
	glAttachShader(ProgramID, fragmentShaderID);
	glLinkProgram(ProgramID);

	// Check the program
	glGetProgramiv(ProgramID, GL_LINK_STATUS, &result);
	glGetProgramiv(ProgramID, GL_INFO_LOG_LENGTH, &infoLogLength);
	std::vector<char> ProgramErrorMessage(glm::max(infoLogLength, int(1)));
	glGetProgramInfoLog(ProgramID, infoLogLength, NULL, &ProgramErrorMessage[0]);
	LOGW("%s\n", &ProgramErrorMessage[0]);

	glDeleteShader(vertexShaderID);
	glDeleteShader(fragmentShaderID);

	n->id = ProgramID;
	n->name = (*newName);

	n->id_worldViewProj = glGetUniformLocation(ProgramID, "WorldViewProj");
	n->id_world = glGetUniformLocation(ProgramID, "World");
	n->id_worldInvTrans = glGetUniformLocation(ProgramID, "WorldInvTrans");
	n->id_eyeVector = glGetUniformLocation(ProgramID, "EyeVector");
	n->id_lightDir = glGetUniformLocation(ProgramID, "LightDir");
	n->id_lightDiff = glGetUniformLocation(ProgramID, "LightDiff");
	n->id_lightSpec = glGetUniformLocation(ProgramID, "LightSpec");
	n->id_lightAmb = glGetUniformLocation(ProgramID, "LightAmb");
	n->id_gloss = glGetUniformLocation(ProgramID, "Gloss");
	n->id_highlight = glGetUniformLocation(ProgramID, "Highlight");
}

void Renderer::ShutdownShader(ShaderID* sid)
{
	glDeleteProgram(sid->id);
}

void Renderer::LoadTexture(const string* filePath, TextureID* id)
{
	id->name = *filePath;
	id->id = SOIL_load_OGL_texture((*filePath).c_str(), SOIL_LOAD_AUTO, SOIL_CREATE_NEW_ID,
		SOIL_FLAG_MIPMAPS | SOIL_FLAG_INVERT_Y | SOIL_FLAG_NTSC_SAFE_RGB | SOIL_FLAG_COMPRESS_TO_DXT);
}

void Renderer::LoadTexture(const string* name, const unsigned char* data, int dataLength, int width, int height, int channels, TextureID* id)
{
	id->name = *name;
	
	glGenTextures(1, (GLuint*)&(id->id));
	glBindTexture(GL_TEXTURE_2D, (GLuint)(id->id));
	glTexImage2D(
		GL_TEXTURE_2D,
		0,
		GL_RGBA,
		width,
		height,
		0,
		GL_RGBA,
		GL_UNSIGNED_BYTE,
		data
		);
}

void Renderer::ShutdownTexture(TextureID* id)
{
	glDeleteTextures(1, (GLuint*)&(id->id));
}