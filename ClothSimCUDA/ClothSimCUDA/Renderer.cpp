#include "Renderer.h"
Renderer* Renderer::instance;

Renderer::Renderer()
{
}

Renderer::Renderer(const Renderer*)
{
}

Renderer::~Renderer()
{
}

Renderer* Renderer::GetInstance()
{
	if (Renderer::instance == nullptr)
	{
		Renderer::instance = new Renderer();
	}

	return Renderer::instance;
}

void Renderer::DestroyInstance()
{
	if (Renderer::instance != nullptr)
		delete Renderer::instance;
}

unsigned int Renderer::Initialize()
{
	unsigned int err = CS_ERR_NONE;

	if (!glfwInit()) return CS_ERR_GLFW_INITIALIZE_FAILED;

	glfwWindowHint(GLFW_SAMPLES, CSSET_GLFW_SAMPLES_VALUE);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	m_window = glfwCreateWindow(CSSET_WINDOW_WIDTH, CSSET_WINDOW_HEIGHT, CSSET_WINDOW_NAME, nullptr, nullptr);
	if (m_window == nullptr)
	{
		glfwTerminate();
		return CS_ERR_WINDOW_FAILED;
	}

	glfwMakeContextCurrent(m_window);
	glewExperimental = true;
	if (glewInit() != GLEW_OK) return CS_ERR_GLEW_INITIALIZE_FAILED;

	m_shaderID = LoadShaders("BasicVertexShader.glsl", "BasicFragmentShader.glsl");

	glClearColor(CSSET_CLEAR_COLORS[0], CSSET_CLEAR_COLORS[1], CSSET_CLEAR_COLORS[2], CSSET_CLEAR_COLORS[3]);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);

	glfwSwapInterval(CSSET_VSYNC_ENALBED);

	return err;
}

unsigned int Renderer::Shutdown()
{
	unsigned int err = CS_ERR_NONE;

	if (m_window != nullptr)
	{
		glfwDestroyWindow(m_window);
		m_window = nullptr;
	}

	glDeleteProgram(m_shaderID);

	return err;
}

unsigned int Renderer::Run()
{
	unsigned int err = CS_ERR_NONE;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUseProgram(m_shaderID);

	// here comes the drawing

	//////

	glfwSwapBuffers(m_window);
	glfwPollEvents();

	return err;
}

GLuint Renderer::GetCurrentShaderID()
{
	return m_shaderID;
}

GLuint Renderer::LoadShaders(const char* vertexFilePath, const char* fragmentFilePath)
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
	std::vector<char> ProgramErrorMessage(glm::max(infoLogLength, int(1)));
	glGetProgramInfoLog(ProgramID, infoLogLength, NULL, &ProgramErrorMessage[0]);
	fprintf(stdout, "%s\n", &ProgramErrorMessage[0]);

	glDeleteShader(vertexShaderID);
	glDeleteShader(fragmentShaderID);

	return ProgramID;
}