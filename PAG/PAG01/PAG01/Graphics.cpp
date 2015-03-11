#include "Graphics.h"


Graphics::Graphics()
{
	m_camera = nullptr;
	m_window = nullptr;
	m_light = nullptr;
}


Graphics::~Graphics()
{
}

bool Graphics::Initialize()
{
	if (!glfwInit())
	{
		fprintf(stderr, "Failed to initialize GLFW\n");
		return false;
	}

	glfwWindowHint(GLFW_SAMPLES, GLFW_SAMPLES_VALUE);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	m_window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_NAME, NULL, NULL);
	if (m_window == NULL)
	{
		fprintf(stderr, "Failed to open GLFW window");
		glfwTerminate();
		return false;
	}

	glfwMakeContextCurrent(m_window);
	glewExperimental = true;
	if (glewInit() != GLEW_OK)
	{
		fprintf(stderr, "Failed to initialize GLFW");
		return false;
	}

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);


	projectionMatrix = perspective(WINDOW_FOV, WINDOW_RATIO, WINDOW_NEAR, WINDOW_FAR);

	m_camera = new Camera();
	if (!m_camera->Initialize()) return false;
	m_camera->Transform(&vec3(0.0f, 0.0f, -4.0f), &vec3(0.0f, 0.0f, 0.0f), &m_camera->GetUp(), &m_camera->GetRight());

	programID = LoadShaders("BasicVertexShader.glsl", "BasicFragmentShader.glsl");

	m_light = new Light(&vec4(0.0f, 0.0f, 1.0f, 0.0f), &(5.0f*vec4(1.0f, 1.0f, 1.0f, 1.0f)), &vec4(1.0f, 1.0f, 1.0f, 1.0f), &vec4(0.0f, 0.0f, 0.0f, 1.0f), 100.0f, programID);

	m_camera->m_eyeVectorID = glGetUniformLocation(programID, "eyeVector");

	m_manager = new MeshManager();
	m_manager->Initialize(programID);
	m_mesh = m_manager->GetMesh(0);
	m_mesh->Transform(&vec3(0.0f, 0.0f, 0.0f), &vec3(-(3.14f / 2.0f), 0.0f, 0.0f), &vec3(0.1f, 0.1f, 0.1f));
	//test = m_manager->GetMesh(0)->GetChildren()->at(0);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

	return true;
}

void Graphics::Shutdown()
{
	if (m_window != nullptr)
	{
		glfwDestroyWindow(m_window);
	}
	if (m_camera != nullptr)
	{
		m_camera->Shutdown();
		delete m_camera;
	}
	if (m_light != nullptr)
	{
		delete m_light;
	}
	if (m_manager != nullptr)
	{
		m_manager->Shutdown();
		delete m_manager;
	}

	glDeleteProgram(programID);
	glfwTerminate();
}

void Graphics::Frame()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUseProgram(programID);

	m_mesh->Transform(
		m_mesh->GetPosition(), 
		&vec3((*m_mesh->GetRotation()).x, (*m_mesh->GetRotation()).y, (*m_mesh->GetRotation()).z + 0.005f),
		m_mesh->GetScale());
		/*
	test->Transform(
		test->GetPosition(),
		&vec3((*test->GetRotation()).x, (*test->GetRotation()).y, (*test->GetRotation()).z + 0.005f),
		test->GetScale());*/

	m_mesh->Draw(&projectionMatrix, m_camera->GetViewMatrix(), &m_camera->GetEyeVector(), m_camera->m_eyeVectorID, m_light);

	glfwSwapBuffers(m_window);
	glfwPollEvents();
}

GLFWwindow* Graphics::GetWindowPtr()
{
	return m_window;
}

Camera* Graphics::GetCameraPtr()
{
	return m_camera;
}

GLuint Graphics::LoadShaders(const char* vertexFilePath, const char* fragmentFilePath)
{
	GLuint vertexShaderID = glCreateShader(GL_VERTEX_SHADER);
	GLuint fragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);

	// Read vs code from file
	string vertexShaderCode;
	ifstream vertexShaderStream(vertexFilePath, ios::in);
	if (vertexShaderStream.is_open())
	{
		string Line = "";
		while (getline(vertexShaderStream, Line))
			vertexShaderCode += "\n" + Line;
		vertexShaderStream.close();
	}

	string fragmentShaderCode;
	ifstream fragmentShaderStream(fragmentFilePath, ios::in);
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
	printf("Compiling shader : %s\n", vertexFilePath);
	char const * VertexSourcePointer = vertexShaderCode.c_str();
	glShaderSource(vertexShaderID, 1, &VertexSourcePointer, NULL);
	glCompileShader(vertexShaderID);

	// Check Vertex Shader
	glGetShaderiv(vertexShaderID, GL_COMPILE_STATUS, &result);
	glGetShaderiv(vertexShaderID, GL_INFO_LOG_LENGTH, &infoLogLength);
	std::vector<char> VertexShaderErrorMessage(infoLogLength);
	glGetShaderInfoLog(vertexShaderID, infoLogLength, NULL, &VertexShaderErrorMessage[0]);
	fprintf(stdout, "%s\n", &VertexShaderErrorMessage[0]);

	// Compile Fragment Shader
	printf("Compiling shader : %s\n", fragmentFilePath);
	char const * FragmentSourcePointer = fragmentShaderCode.c_str();
	glShaderSource(fragmentShaderID, 1, &FragmentSourcePointer, NULL);
	glCompileShader(fragmentShaderID);

	// Check Fragment Shader
	glGetShaderiv(fragmentShaderID, GL_COMPILE_STATUS, &result);
	glGetShaderiv(fragmentShaderID, GL_INFO_LOG_LENGTH, &infoLogLength);
	std::vector<char> FragmentShaderErrorMessage(infoLogLength);
	glGetShaderInfoLog(fragmentShaderID, infoLogLength, NULL, &FragmentShaderErrorMessage[0]);
	fprintf(stdout, "%s\n", &FragmentShaderErrorMessage[0]);

	// Link the program
	fprintf(stdout, "Linking program\n");
	GLuint ProgramID = glCreateProgram();
	glAttachShader(ProgramID, vertexShaderID);
	glAttachShader(ProgramID, fragmentShaderID);
	glLinkProgram(ProgramID);

	// Check the program
	glGetProgramiv(ProgramID, GL_LINK_STATUS, &result);
	glGetProgramiv(ProgramID, GL_INFO_LOG_LENGTH, &infoLogLength);
	std::vector<char> ProgramErrorMessage(max(infoLogLength, int(1)));
	glGetProgramInfoLog(ProgramID, infoLogLength, NULL, &ProgramErrorMessage[0]);
	fprintf(stdout, "%s\n", &ProgramErrorMessage[0]);

	glDeleteShader(vertexShaderID);
	glDeleteShader(fragmentShaderID);

	return ProgramID;
}
