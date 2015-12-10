#include "Renderer.h"

Renderer::Renderer()
{
	m_initialized = false;
}

Renderer::Renderer(const Renderer*)
{
}

Renderer::~Renderer()
{
}


unsigned int Renderer::Initialize()
{
	if (m_initialized)
		return CS_ERR_NONE;
	unsigned int err = CS_ERR_NONE;

	// initialize OpenGL ES and EGL

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
	context = eglCreateContext(display, config, NULL, attribsContext);

	if (eglMakeCurrent(display, surface, surface, context) == EGL_FALSE) {
		LOGW("Unable to eglMakeCurrent");
		return -1;
	}

	eglQuerySurface(display, surface, EGL_WIDTH, &w);
	eglQuerySurface(display, surface, EGL_HEIGHT, &h);

	m_screenRatio = (float)w / (float)h;

	engine->display = display;
	engine->context = context;
	engine->surface = surface;
	engine->width = w;
	engine->height = h;

	// Shaders Loading
	
	m_basicShader = ResourceManager::GetInstance()->LoadShader(&SN_BASIC);
	m_wireframeShader = ResourceManager::GetInstance()->LoadShader(&SN_WIREFRAME);
	m_fontShader = ResourceManager::GetInstance()->LoadShader(&SN_FONT);
	m_shaderID = m_basicShader;

	m_mode = BASIC;

	//////////// options go here

	glClearColor(CSSET_CLEAR_COLORS[0], CSSET_CLEAR_COLORS[1], CSSET_CLEAR_COLORS[2], CSSET_CLEAR_COLORS[3]);
	glEnable(GL_DEPTH_TEST);

	if (CSSET_BACKFACE_CULLING)
	{
		glEnable(GL_CULL_FACE);
	}
	glCullFace(GL_BACK);
	glFrontFace(GL_CCW);

	glEnable(GL_DITHER);
	//glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_FASTEST);
	/*
	glEnable(GL_STENCIL);
	glStencilFunc(GL_LEQUAL, 0, 0xFF);
	glStencilOp(GL_REPLACE, GL_KEEP, GL_KEEP);
	*/
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	

	glDepthFunc(GL_LEQUAL);

	/////////////////////////

	m_initialized = true;
	return err;
}

unsigned int Renderer::Shutdown()
{
	if (!m_initialized)
		return CS_ERR_NONE;

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

	m_initialized = false;
	return err;
}

unsigned int Renderer::Run()
{
	unsigned int err = CS_ERR_NONE;
	Engine* engine = System::GetInstance()->GetEngineData();

	if (engine->display == NULL || engine->context == NULL || engine->surface == NULL) 
	{
		// No display.
		return CS_ERR_UNKNOWN;
	}

	if (m_resizeNeeded)
	{
		ResizeViewport();
		m_resizeNeeded = false;
	}	

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Drawing models with basic shader

	if (m_mode == BASIC)
	{
		m_shaderID = m_basicShader;
		glUseProgram(m_shaderID->id);
	}
	
	else if (m_mode == WIREFRAME) // Drawing wireframe
	{
		m_shaderID = m_wireframeShader;
		glUseProgram(m_shaderID->id);
	}

	else if (m_mode == BASIC_WIREFRAME) // Drawing wireframe with fill (no specular)
	{
		m_shaderID = m_basicShader;	////!!!!!!!
		glUseProgram(m_shaderID->id);
	}

	System::GetInstance()->GetCurrentScene()->DrawObjects();

	// Drawing GUI
	m_shaderID = m_fontShader;
	glUseProgram(m_shaderID->id);
	System::GetInstance()->GetCurrentScene()->DrawGUI();

	EGLBoolean res = eglSwapBuffers(engine->display, engine->surface);

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

float Renderer::GetScreenRatio()
{
	return m_screenRatio;
}

bool Renderer::GetInitialized()
{
	return m_initialized;
}



void Renderer::LoadKernel(const string * filePath, const string * newName, KernelID * data)
{
	GLuint vID = glCreateShader(GL_VERTEX_SHADER);

	// Read vs code from file
	char *vertexShaderCode;
	vertexShaderCode = LoadKernelFromAssets(filePath);

	GLint result = 0;
	int infoLogLength = 0;

	// Compile Vertex Shader
	LOGI("Compiling kernel : %s\n", filePath->c_str());
	char const * VertexSourcePointer = vertexShaderCode;
	glShaderSource(vID, 1, &VertexSourcePointer, NULL);
	glCompileShader(vID);

	// Check Vertex Shader
	glGetShaderiv(vID, GL_COMPILE_STATUS, &result);
	glGetShaderiv(vID, GL_INFO_LOG_LENGTH, &infoLogLength);
	vector<char> VertexShaderErrorMessage(infoLogLength);
	glGetShaderInfoLog(vID, infoLogLength, NULL, &VertexShaderErrorMessage[0]);
	LOGW("%s : %s\n", filePath->c_str(), &VertexShaderErrorMessage[0]);

	// Link the program
	LOGI("Linking program\n");
	GLuint ProgramID = glCreateProgram();
	glAttachShader(ProgramID, vID);
	glLinkProgram(ProgramID);

	// Check the program
	glGetProgramiv(ProgramID, GL_LINK_STATUS, &result);
	glGetProgramiv(ProgramID, GL_INFO_LOG_LENGTH, &infoLogLength);
	vector<char> ProgramErrorMessage(glm::max(infoLogLength, int(1)));
	glGetProgramInfoLog(ProgramID, infoLogLength, NULL, &ProgramErrorMessage[0]);
	LOGW("%s\n", &ProgramErrorMessage[0]);

	glDeleteShader(vID);
	delete vertexShaderCode;

	data->id = ProgramID;
	data->name = *newName;
}

void Renderer::LoadShaders(const string* vertexFilePath, const string* fragmentFilePath, const string* newName, ShaderID* n)
{
	GLuint vertexShaderID = glCreateShader(GL_VERTEX_SHADER);
	GLuint fragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);


	// Read vs code from file
	char *vertexShaderCode, *fragmentShaderCode;
	vertexShaderCode = LoadShaderFromAssets(vertexFilePath);
	fragmentShaderCode = LoadShaderFromAssets(fragmentFilePath);

	GLint result = 500;
	int infoLogLength = 10;

	// Compile Vertex Shader
	LOGI("Compiling shader : %s\n", vertexFilePath->c_str());
	char const * VertexSourcePointer = vertexShaderCode;
	glShaderSource(vertexShaderID, 1, &VertexSourcePointer, NULL);
	glCompileShader(vertexShaderID);

	// Check Vertex Shader
	glGetShaderiv(vertexShaderID, GL_COMPILE_STATUS, &result);
	glGetShaderiv(vertexShaderID, GL_INFO_LOG_LENGTH, &infoLogLength);
	vector<char> VertexShaderErrorMessage(infoLogLength);
	glGetShaderInfoLog(vertexShaderID, infoLogLength, NULL, &VertexShaderErrorMessage[0]);
	LOGW("%s : %s\n", vertexFilePath->c_str(), &VertexShaderErrorMessage[0]);

	// Compile Fragment Shader
	LOGI("Compiling shader : %s\n", fragmentFilePath->c_str());
	char const * FragmentSourcePointer = fragmentShaderCode;
	glShaderSource(fragmentShaderID, 1, &FragmentSourcePointer, NULL);
	glCompileShader(fragmentShaderID);

	// Check Fragment Shader
	glGetShaderiv(fragmentShaderID, GL_COMPILE_STATUS, &result);
	glGetShaderiv(fragmentShaderID, GL_INFO_LOG_LENGTH, &infoLogLength);
	vector<char> FragmentShaderErrorMessage(infoLogLength);
	glGetShaderInfoLog(fragmentShaderID, infoLogLength, NULL, &FragmentShaderErrorMessage[0]);
	LOGW("%s : %s\n", fragmentFilePath->c_str(), &FragmentShaderErrorMessage[0]);

	// Link the program
	LOGI("Linking program\n");
	GLuint ProgramID = glCreateProgram();
	glAttachShader(ProgramID, vertexShaderID);
	glAttachShader(ProgramID, fragmentShaderID);
	glLinkProgram(ProgramID);

	// Check the program
	glGetProgramiv(ProgramID, GL_LINK_STATUS, &result);
	glGetProgramiv(ProgramID, GL_INFO_LOG_LENGTH, &infoLogLength);
	vector<char> ProgramErrorMessage(glm::max(infoLogLength, int(1)));
	glGetProgramInfoLog(ProgramID, infoLogLength, NULL, &ProgramErrorMessage[0]);
	LOGW("%s\n", &ProgramErrorMessage[0]);

	glDeleteShader(vertexShaderID);
	glDeleteShader(fragmentShaderID);
	delete vertexShaderCode;
	delete fragmentShaderCode;

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

char* Renderer::LoadShaderFromAssets(const string * path)
{
	string prefix = "shaders/";
	string suffix = ".glsl";
	string fPath = prefix + *path + suffix;

	// fuck you, assetmanager
	AAssetManager* mgr = System::GetInstance()->GetEngineData()->app->activity->assetManager;

	AAsset* shaderAsset = AAssetManager_open(mgr, fPath.c_str(), AASSET_MODE_UNKNOWN);
	unsigned int length = AAsset_getLength(shaderAsset);

	char * code = new char[length + 1];

	AAsset_read(shaderAsset, (void*)code, length);
	code[length] = '\0';
	return code;
}

char* Renderer::LoadKernelFromAssets(const string * path)
{
	string prefix = "kernels/";
	string suffix = ".glsl";
	string fPath = prefix + *path + suffix;

	// fuck you, assetmanager
	AAssetManager* mgr = System::GetInstance()->GetEngineData()->app->activity->assetManager;

	AAsset* shaderAsset = AAssetManager_open(mgr, fPath.c_str(), AASSET_MODE_UNKNOWN);
	unsigned int length = AAsset_getLength(shaderAsset);

	char * code = new char[length + 1];

	AAsset_read(shaderAsset, (void*)code, length);
	code[length] = '\0';
	return code;
}

inline void Renderer::ResizeViewport()
{
	Engine* engine = System::GetInstance()->GetEngineData();

	glViewport(0, 0, engine->width, engine->height);
}

void Renderer::ShutdownShader(ShaderID* sid)
{
	glDeleteProgram(sid->id);
}

void Renderer::LoadTexture(const string* filePath, TextureID* id)
{
	AAssetManager* mgr = System::GetInstance()->GetEngineData()->app->activity->assetManager;
	AAsset* textureAsset = AAssetManager_open(mgr, filePath->c_str(), AASSET_MODE_UNKNOWN);
	unsigned int length = AAsset_getLength(textureAsset);

	unsigned char * bitmap = new unsigned char[length];
	AAsset_read(textureAsset, (void*)bitmap, length);

	id->name = *filePath;
	id->id = SOIL_load_OGL_texture_from_memory(bitmap, length, SOIL_LOAD_AUTO, SOIL_CREATE_NEW_ID,
		SOIL_FLAG_MIPMAPS | SOIL_FLAG_INVERT_Y | SOIL_FLAG_NTSC_SAFE_RGB);

	++length;
	//id->id = SOIL_load_OGL_texture((*filePath).c_str(), SOIL_LOAD_AUTO, SOIL_CREATE_NEW_ID,
		//SOIL_FLAG_MIPMAPS | SOIL_FLAG_INVERT_Y | SOIL_FLAG_NTSC_SAFE_RGB | SOIL_FLAG_COMPRESS_TO_DXT);
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

void Renderer::AHandleResize(ANativeActivity * activity, ANativeWindow * window)
{
	int32_t w, h;
	w = ANativeWindow_getWidth(window);
	h = ANativeWindow_getHeight(window);

	Engine* engine = System::GetInstance()->GetEngineData();
	engine->width = w;
	engine->height = h;

	Renderer::GetInstance()->m_resizeNeeded = true;

	if(System::GetInstance()->GetCurrentScene() != nullptr)
		System::GetInstance()->GetCurrentScene()->FlushDimensions();

	LOGI("Renderer: Resize scheduled %dx%d", w, h);
}
