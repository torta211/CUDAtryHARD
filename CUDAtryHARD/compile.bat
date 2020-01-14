:: nvcc -O3 texture2.cu -o texture2.exe
:: nvcc -O3 main.cu -I"D:\glfw-3.3\include" -I"C:\Users\All Users\NVIDIA Corporation\CUDA Samples\v10.0\common\inc" -L"C:\Users\All Users\NVIDIA Corporation\CUDA Samples\v10.0\common\lib\x64" -o gl.exe -lglew64 -lopengl32 -lglfw3 -lkernel32 -luser32 -lgdi32

nvcc -O3 nbody_gl.cu -I"D:\glfw-3.3\include" -I"C:\Users\All Users\NVIDIA Corporation\CUDA Samples\v10.0\common\inc" -L"C:\Users\All Users\NVIDIA Corporation\CUDA Samples\v10.0\common\lib\x64" -o gl.exe -lglew64 -lopengl32 -lglfw3 -lkernel32 -luser32 -lgdi32