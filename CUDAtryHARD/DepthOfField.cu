#include <GL_E/glew.h>
#include <GLFW/glfw3.h>

#include <iostream>


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

    std::cout << "Entering render loop\n";
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
    }
}
                              











