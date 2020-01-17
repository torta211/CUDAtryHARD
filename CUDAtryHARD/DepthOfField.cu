#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN 1
#include <windows.h>
#else
#endif

#include <GL_E/glew.h>
#include <GLFW/glfw3.h>
#ifdef _WIN32
#else
#include <GL/glx.h>
#include <GL/glext.h>
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>

#include <string>
#include <algorithm>
#include <vector>
#include <random>
#include <iostream>

__host__ __device__ float sq(float x) { return x * x; }
__host__ __device__ float cube(float x) { return x * x * x; }

struct Particle { float x, y, z, m; };

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
template<typename fptr_type>
fptr_type load_extension_pointer(const char* name) { return reinterpret_cast<fptr_type>(wglGetProcAddress(name)); }
#else
template<typename fptr_type>
fptr_type load_extension_pointer(const char* name) { return reinterpret_cast<fptr_type>(glXGetProcAddressARB((const GLubyte*)name)); }
#endif

static void error_callback(int error, const char* description)
{
	std::cout << "Error: " << description << "\n";
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
	{
		glfwSetWindowShouldClose(window, GLFW_TRUE);
	}
}

static inline const char* glErrorToString(GLenum err)
{
#define CASE_RETURN_MACRO(arg) case arg: return #arg
	switch (err)
	{
		CASE_RETURN_MACRO(GL_NO_ERROR);
		CASE_RETURN_MACRO(GL_INVALID_ENUM);
		CASE_RETURN_MACRO(GL_INVALID_VALUE);
		CASE_RETURN_MACRO(GL_INVALID_OPERATION);
		CASE_RETURN_MACRO(GL_OUT_OF_MEMORY);
		CASE_RETURN_MACRO(GL_STACK_UNDERFLOW);
		CASE_RETURN_MACRO(GL_STACK_OVERFLOW);
#ifdef GL_INVALID_FRAMEBUFFER_OPERATION
		CASE_RETURN_MACRO(GL_INVALID_FRAMEBUFFER_OPERATION);
#endif
	default: break;
	}
#undef CASE_RETURN_MACRO
	return "*UNKNOWN*";
}

bool checkGLError(const char* msg = "")
{
	GLenum gl_error = glGetError();
	if (gl_error != GL_NO_ERROR)
	{
		std::cout << "GL error: " << glErrorToString(gl_error) << " msg: " << msg << "\n";
		return false;
	}
	return true;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	glViewport(0, 0, width, height);
}

int main(void)
{
	//Create window:
	int width = 640;
	int height = 480;

	glfwSetErrorCallback(error_callback);
	if (!glfwInit()) { return -1; }

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	GLFWwindow* window = glfwCreateWindow(width, height, "Simple example", NULL, NULL);
	if (!window)
	{
		glfwTerminate();
		return -1;
	}
	glfwSetKeyCallback(window, key_callback);
	glfwMakeContextCurrent(window);
	glfwSwapInterval(0);

	glewExperimental = GL_TRUE;
	GLenum err = glewInit();
	glGetError();
	if (err != GLEW_OK)
	{
		std::cout << "glewInit failed: " << glewGetErrorString(err);
		return -1;
	}
	glEnable(GL_PROGRAM_POINT_SIZE);
	glEnable(GL_DEPTH_TEST);

	glViewport(0, 0, width, height);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

	//verify the openGL version we got:
	{
		int p = glfwGetWindowAttrib(window, GLFW_OPENGL_PROFILE);
		std::string version = glfwGetVersionString();
		std::string opengl_profile = "";
		if (p == GLFW_OPENGL_COMPAT_PROFILE) { opengl_profile = "OpenGL Compatibility Profile"; }
		else if (p == GLFW_OPENGL_CORE_PROFILE) { opengl_profile = "OpenGL Core Profile"; }
		std::cout << "GLFW version: " << version << "\n";
		std::cout << "GLFW OpenGL profile: " << opengl_profile << "\n";

		std::cout << "OpenGL: GL version: " << glGetString(GL_VERSION) << "\n";
		std::cout << "OpenGL: GLSL version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << "\n";
		std::cout << "OpenGL: Vendor: " << glGetString(GL_VENDOR) << "\n";

		std::cout << "GLEW: Glew version: " << glewGetString(GLEW_VERSION) << "\n";
	}

	//Generate vertices
	const int N = 224;

	std::vector<Particle> particles(N);
	float dx = 0.5f;
	float dz = 0.04f;
	for (int i = 0; i < N; i += 8)
	{
		particles[i    ] = Particle{ -5.0f + i / 8 * dx, -5.25f, -0.2f + i / 8 * dz, 10000.0f };
		particles[i + 1] = Particle{ -5.0f + i / 8 * dx, -3.75f, -0.2f + i / 8 * dz, 10000.0f };
		particles[i + 2] = Particle{ -5.0f + i / 8 * dx, -2.25f, -0.2f + i / 8 * dz, 10000.0f };
		particles[i + 3] = Particle{ -5.0f + i / 8 * dx, -0.75f, -0.2f + i / 8 * dz, 10000.0f };
		particles[i + 4] = Particle{ -5.0f + i / 8 * dx,  0.75f, -0.2f + i / 8 * dz, 10000.0f };
		particles[i + 5] = Particle{ -5.0f + i / 8 * dx,  2.25f, -0.2f + i / 8 * dz, 10000.0f };
		particles[i + 6] = Particle{ -5.0f + i / 8 * dx,  3.75f, -0.2f + i / 8 * dz, 10000.0f };
		particles[i + 7] = Particle{ -5.0f + i / 8 * dx,  5.25f, -0.2f + i / 8 * dz, 10000.0f };
	}

	// quad vertices for screen
	float quadVerticesArray[] = {
		-1.0f,  1.0f, 0.0f, 1.0f,
		-1.0f, -1.0f, 0.0f, 0.0f,
		 1.0f, -1.0f, 1.0f, 0.0f,

		-1.0f,  1.0f, 0.0f, 1.0f,
		 1.0f, -1.0f, 1.0f, 0.0f,
		 1.0f,  1.0f, 1.0f, 1.0f
	};
	std::vector<float> quadVertices(quadVerticesArray, quadVerticesArray + sizeof(quadVerticesArray) / sizeof(float));

	//Compile shaders:
	auto load_and_compile_shader = [](auto shader_type, std::string const& path)->GLuint
	{
		std::basic_string<GLchar> string;

		if (path.size() != 0)
		{
			std::basic_ifstream<GLchar> file(path);
			if (!file.is_open()) { std::cout << "Cannot open shader file: " << path << "\n"; return 0; }
			string = std::basic_string<GLchar>(std::istreambuf_iterator<GLchar>(file), (std::istreambuf_iterator<GLchar>()));
		}
		else
		{
			//string = std::basic_string<GLchar>{ shader_type == GL_VERTEX_SHADER ? vertex_shader_str : fragment_shader_str };
			return 0;
		}
		const GLchar* tmp = string.c_str();

		auto shaderObj = glCreateShader(shader_type);
		if (!checkGLError()) { return 0; }

		GLint gl_status = 0;
		glShaderSource(shaderObj, (GLsizei)1, &tmp, NULL);
		glCompileShader(shaderObj);
		glGetShaderiv(shaderObj, GL_COMPILE_STATUS, &gl_status);

		if (!gl_status)
		{
			GLint log_size;
			glGetShaderiv(shaderObj, GL_INFO_LOG_LENGTH, &log_size);
			std::basic_string<GLchar> log(log_size, ' ');
			glGetShaderInfoLog(shaderObj, log_size, NULL, &(*log.begin()));
			std::cout << "Failed to compile shader: " << std::endl << log << std::endl;
		}
		else
		{
			std::cout << "Shader " << path << " compiled successfully\n";
		}

		return shaderObj;
	};

	GLuint vertexShaderObj = load_and_compile_shader(GL_VERTEX_SHADER, "vertex.glsl");
	GLuint fragmentShaderObj = load_and_compile_shader(GL_FRAGMENT_SHADER, "fragment.glsl");
	if (!vertexShaderObj && !fragmentShaderObj) { std::cout << "Failed to load and compile shaders\n"; return -1; }

	GLuint glProgram = glCreateProgram();
	{
		glAttachShader(glProgram, vertexShaderObj);
		glAttachShader(glProgram, fragmentShaderObj);
		glLinkProgram(glProgram);

		GLint gl_status = 0;
		glGetProgramiv(glProgram, GL_LINK_STATUS, &gl_status);
		if (!gl_status)
		{
			char temp[256];
			glGetProgramInfoLog(glProgram, 256, 0, temp);
			std::cout << "Failed to link program: " << temp << std::endl;
			glDeleteProgram(glProgram);
		}
		else { std::cout << "Shaders linked successfully\n"; }
		if (!checkGLError()) { return -1; }
	}

	GLuint vertexTextureShaderObj = load_and_compile_shader(GL_VERTEX_SHADER, "textureVertexShader.glsl");
	GLuint fragmentTextureShaderObj = load_and_compile_shader(GL_FRAGMENT_SHADER, "textureFragmentShader.glsl");
	if (!vertexTextureShaderObj && !fragmentTextureShaderObj) { std::cout << "Failed to load and compile shaders\n"; return -1; }

	GLuint textureShaderProgram = glCreateProgram();
	{
		glAttachShader(textureShaderProgram, vertexTextureShaderObj);
		glAttachShader(textureShaderProgram, fragmentTextureShaderObj);
		glLinkProgram(textureShaderProgram);

		GLint gl_status = 0;
		glGetProgramiv(textureShaderProgram, GL_LINK_STATUS, &gl_status);
		if (!gl_status)
		{
			char temp[256];
			glGetProgramInfoLog(textureShaderProgram, 256, 0, temp);
			std::cout << "Failed to link program: " << temp << std::endl;
			glDeleteProgram(textureShaderProgram);
		}
		else { std::cout << "Shaders linked successfully\n"; }

		if (!checkGLError()) { return -1; }
	}

	//Create buffers (quad for texture):
	GLuint quadvbo;
	GLuint quadvao;

	//Create buffer ID
	glGenBuffers(1, &quadvbo);
	if (!checkGLError("glGenBuffers")) { return -1; }

	// Select the GL Buffer as the active one: 
	glBindBuffer(GL_ARRAY_BUFFER, quadvbo);
	if (!checkGLError("glBindBuffer")) { return -1; }

	// Allocate memory for the buffer:
	glBufferData(GL_ARRAY_BUFFER, quadVertices.size() * sizeof(float), NULL, GL_STATIC_DRAW);
	if (!checkGLError("glBufferData")) { return -1; }

	// Upload data to the buffer:
	glBufferSubData(GL_ARRAY_BUFFER, 0, quadVertices.size() * sizeof(float), quadVertices.data());
	if (!checkGLError("glBufferSubData")) { return -1; }

	// Create and activate the Vertex Array Object
	// these API functions are missing on windows from the CUDA SDK glew, so we load them manually:
	typedef void (*Fnt_GenVertexArrays) (GLsizei n, GLuint* arrays);
	typedef void (*Fnt_BindVertexArray) (GLuint array);
	Fnt_GenVertexArrays pglGenVertexArrays = load_extension_pointer<Fnt_GenVertexArrays>("glGenVertexArrays");
	Fnt_BindVertexArray pglBindVertexArray = load_extension_pointer<Fnt_BindVertexArray>("glBindVertexArray");

	pglGenVertexArrays(1, &quadvao); if (!checkGLError("glGenVertexArrays")) { return -1; }
	pglBindVertexArray(quadvao);	   if (!checkGLError("glBindVertexArray")) { return -1; }

	// Register buffers into the VAO
	glBindBuffer(GL_ARRAY_BUFFER, quadvbo);
	if (!checkGLError("glBindBuffer(geo)")) { return -1; }
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
	if (!checkGLError("glVertexAttribPointer(geo)")) { return -1; }

	glEnableVertexAttribArray(0);
	if (!checkGLError("glEnableVertexAttribArray(0)")) { return -1; }

	//Create buffers (for particle data):
	GLuint glbuffer;
	GLuint glvao;

	//Create buffer ID
	glGenBuffers(1, &glbuffer);
	if (!checkGLError("glGenBuffers")) { return -1; }

	// Select the GL Buffer as the active one: 
	glBindBuffer(GL_ARRAY_BUFFER, glbuffer);
	if (!checkGLError("glBindBuffer")) { return -1; }

	// Allocate memory for the buffer:
	glBufferData(GL_ARRAY_BUFFER, particles.size() * sizeof(Particle), NULL, GL_STATIC_DRAW);
	if (!checkGLError("glBufferData")) { return -1; }

	// Upload data to the buffer:
	glBufferSubData(GL_ARRAY_BUFFER, 0, particles.size() * sizeof(Particle), particles.data());
	if (!checkGLError("glBufferSubData")) { return -1; }

	pglGenVertexArrays(1, &glvao); if (!checkGLError("glGenVertexArrays")) { return -1; }
	pglBindVertexArray(glvao);	   if (!checkGLError("glBindVertexArray")) { return -1; }

	// Register buffers into the VAO
	glBindBuffer(GL_ARRAY_BUFFER, glbuffer);
	if (!checkGLError("glBindBuffer(geo)")) { return -1; }
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(Particle), (GLvoid*)0);
	if (!checkGLError("glVertexAttribPointer(geo)")) { return -1; }

	glEnableVertexAttribArray(0);
	if (!checkGLError("glEnableVertexAttribArray(0)")) { return -1; }

	// create framebuffer
	unsigned int fbo;
	glGenFramebuffers(1, &fbo);

	// create a color attachment texture
	unsigned int textureColorbuffer;
	glGenTextures(1, &textureColorbuffer);
	glBindTexture(GL_TEXTURE_2D, textureColorbuffer);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	// create a renderbuffer object for depth and stancil attachment, because for now, we won't sample these
	unsigned int rbo;
	glGenRenderbuffers(1, &rbo);
	glBindRenderbuffer(GL_RENDERBUFFER, rbo);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height);

	// attach the texture and renderbuffer
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo); // write operations
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureColorbuffer, 0);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo);

	// verify if the framebuffer is complete
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE)
	{
		std::cout << "Framebuffer complete\n";
	}

	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
	glBindTexture(GL_TEXTURE_2D, textureColorbuffer);
	int w, h, nrChannels;
	unsigned char* data = stbi_load("map.png", &w, &h, &nrChannels, 0);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
	glGenerateMipmap(GL_TEXTURE_2D);
	stbi_image_free(data);
	std::cout << "Entering render loop\n";
	while (!glfwWindowShouldClose(window))
	{
		glClearColor(0, 0, 0, 1);
		glClear(GL_COLOR_BUFFER_BIT);

		glUseProgram(textureShaderProgram);
		pglBindVertexArray(quadvao);
		glDrawArrays(GL_TRIANGLES, 0, 6);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}
	glfwDestroyWindow(window);
	glfwTerminate();
	return 0;
}