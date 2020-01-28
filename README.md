# CUDAtryHARD
GPGPPUprog HW

Visual studio 2019 working setup

1.) uninstall everything that is related to nvidia
2.) install CUDA, (reinstall any additional drivers optionally)
3.) create a Visual Studio project from the CUDA template
4.) download glfw and glew windows prebuilt binaries
5.) include the include folders
6.) add the lib folders of 64 bit version to the project
7.) Linker->Input->additional dependencies: opengl32.lib, glew32.lib, glfw3.lib
8.) copy glew/bin/x64/glew32.dll to the project folder

DONE

DEMO:
https://drive.google.com/open?id=1j-BS3pyLDXrW1oK-5qYeqmRM2ZWVsBh9

DEPENDENCIES:

GLFW
GLAD
GLM
