#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <random>
#include <chrono>
#include <iostream>

struct color { float r, g, b, a; };

struct rawcolor { unsigned char r, g, b, a; };

std::vector<color> cpu_create_integral(std::vector<color>& orig_image, int w, int h)
{
    // 4 nested for loops = runs forever (many minutes)
    // this is a more efficient algorithm, but still not the best
    std::vector<color> integral_image(orig_image.size());
    // 0,0 element
    integral_image[0 * w + 0] = orig_image[0 * w + 0];
    // first row
    for (int col = 1; col < w; col++)
    {
        float r = integral_image[0 * w + col - 1].r + orig_image[0 * w + col].r;
        float g = integral_image[0 * w + col - 1].g + orig_image[0 * w + col].g;
        float b = integral_image[0 * w + col - 1].b + orig_image[0 * w + col].b;
        float a = integral_image[0 * w + col - 1].a + orig_image[0 * w + col].a;
        integral_image[0 * w + col] = color{r, g, b, a};
    }
    // first column
    for (int row = 1; row < h; row++)
    {
        float r = integral_image[(row - 1) * w + 0].r + orig_image[row * w + 0].r;
        float g = integral_image[(row - 1) * w + 0].g + orig_image[row * w + 0].g;
        float b = integral_image[(row - 1) * w + 0].b + orig_image[row * w + 0].b;
        float a = integral_image[(row - 1) * w + 0].a + orig_image[row * w + 0].a;
        integral_image[row * w + 0] = color{ r, g, b, a };
    }
    // rest of the image
    for (int col = 1; col < w; col++)
    {
        for (int row = 1; row < h; row++)
        {
            int curr = row * w + col;
            int left = row * w + col - 1;
            int uppe = (row - 1) * w + col;
            int uple = (row - 1) * w + col - 1;

            float r = integral_image[left].r + integral_image[uppe].r - integral_image[uple].r + orig_image[curr].r;
            float g = integral_image[left].g + integral_image[uppe].g - integral_image[uple].g + orig_image[curr].g;
            float b = integral_image[left].b + integral_image[uppe].b - integral_image[uple].b + orig_image[curr].b;
            float a = integral_image[left].a + integral_image[uppe].a - integral_image[uple].a + orig_image[curr].a;

            integral_image[curr] = color{ r, g, b, a };
        }
    }
    return integral_image;
}

// will not use CUDA textures this time (It could be interesting to expreiment with that too)
__global__ void gpu_blur_integral(float4* render_out, float4* integral_in, float* depth_in, int W, int H, float sharp_dist)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    int kernel_radius = 10 * abs(depth_in[y * W + x] - sharp_dist) + 1;

    // corner1, corner2
    // corner3, corner4
    int corner1_x = x - kernel_radius > 0 ? x - kernel_radius : 0;
    int corner1_y = y - kernel_radius > 0 ? y - kernel_radius : 0;
    float4 corner1 = integral_in[corner1_y * W + corner1_x];

    int corner2_x = x + kernel_radius - 1 < W ? x + kernel_radius - 1 : W - 1;
    int corner2_y = y - kernel_radius > 0 ? y - kernel_radius : 0;
    float4 corner2 = integral_in[corner2_y * W + corner2_x];

    int corner3_x = x - kernel_radius > 0 ? x - kernel_radius : 0;
    int corner3_y = y + kernel_radius - 1 < H ? y + kernel_radius - 1 : H - 1;
    float4 corner3 = integral_in[corner3_y * W + corner3_x];

    int corner4_x = x + kernel_radius - 1 < W ? x + kernel_radius - 1: W - 1;
    int corner4_y = y + kernel_radius - 1 < H ? y + kernel_radius - 1: H - 1;
    float4 corner4 = integral_in[corner4_y * W + corner4_x];

    float blurred_val_r = corner4.x - corner3.x - corner2.x + corner1.x;
    float blurred_val_g = corner4.y - corner3.y - corner2.y + corner1.y;
    float blurred_val_b = corner4.z - corner3.z - corner2.z + corner1.z;
    float blurred_val_a = corner4.w - corner3.w - corner2.w + corner1.w;
    blurred_val_r /= (corner2_x - corner1_x) * (corner3_y - corner1_y);
    blurred_val_g /= (corner2_x - corner1_x) * (corner3_y - corner1_y);
    blurred_val_b /= (corner2_x - corner1_x) * (corner3_y - corner1_y);
    blurred_val_a /= (corner2_x - corner1_x) * (corner3_y - corner1_y);

    render_out[y * W + x] = float4{ blurred_val_r, blurred_val_g, blurred_val_b, blurred_val_a};
}

int main()
{
    static const std::string input_image_filename = "church.png";
    static const std::string input__depth_filename = "church_depth.png";
    static const std::string output_filename1 = "gpu_out1.jpg";
    static const std::string output_filename2 = "gpu_out2.jpg";

    static const int block_size = 32;

    int w = 0;//width
    int h = 0;//height
    int ch = 0;//number of components
    
    rawcolor* data0 = reinterpret_cast<rawcolor*>(stbi_load(input_image_filename.c_str(), &w, &h, &ch, 4 /* we expect 4 channels */));
    if (!data0)
    {
        std::cout << "Error: could not open input file: " << input_image_filename << "\n";
        return -1;
    }
    else
    {
        std::cout << "Image (" << input_image_filename << ") opened successfully. Width x Height x Components = " << w << " x " << h << " x " << ch << "\n";
    }

    int w_d = 0;//width
    int h_d = 0;//height
    int ch_d = 0;//number of components

    unsigned char* data1 = reinterpret_cast<unsigned char*>(stbi_load(input__depth_filename.c_str(), &w_d, &h_d, &ch_d, 1 /* we expect 1 channel */));
    if (!data1)
    {
        std::cout << "Error: could not open input file: " << input__depth_filename << "\n";
        return -1;
    }
    else if (w != w_d || h != h_d)
    {
        std::cout << "Error: color and depth image size mismatch\n";
        return -1;
    }
    else
    {
        std::cout << "Image (" << input__depth_filename << ") opened successfully. Width x Height x Components = " << w_d << " x " << h_d << " x " << ch_d << "\n";
    }

    std::vector<color> input(w * h);
    std::vector<float> input_depth(w * h);
    std::vector<color> output(w * h);

    std::transform(data0, data0 + w * h, input.begin(), [](rawcolor c) { return color{ c.r / 255.0f, c.g / 255.0f, c.b / 255.0f, c.a / 255.0f }; });
    stbi_image_free(data0);

    std::transform(data1, data1 + w * h, input_depth.begin(), [](unsigned char c) { return c / 255.0f; });
    stbi_image_free(data1);

    std::vector<color> input_integral = cpu_create_integral(input, w, h);

    std::cout << "Enter the focus distance (scaled to 0 - 1):\n";
    float focus_distance = 0.0f;
    std::cin >> focus_distance;

    //GPU version using buffers:
    float dt = 0.0f;//milliseconds
    {
        float4* pInput = nullptr;
        float4* pOutput = nullptr;
        float* pDepth = nullptr;

        cudaEvent_t evt[2];
        for (auto& e : evt) { cudaEventCreate(&e); }

        cudaError_t err = cudaSuccess;
        err = cudaMalloc((void**)&pInput, w * h * sizeof(color));
        if (err != cudaSuccess) { std::cout << "Error allocating CUDA memory: " << cudaGetErrorString(err) << "\n"; return -1; }

        err = cudaMalloc((void**)&pOutput, w * h * sizeof(color));
        if (err != cudaSuccess) { std::cout << "Error allocating CUDA memory: " << cudaGetErrorString(err) << "\n"; return -1; }

        err = cudaMalloc((void**)&pDepth, w * h * sizeof(float));
        if (err != cudaSuccess) { std::cout << "Error allocating CUDA memory: " << cudaGetErrorString(err) << "\n"; return -1; }

        err = cudaMemcpy(pInput, input_integral.data(), w * h * sizeof(color), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) { std::cout << "Error copying memory to device: " << cudaGetErrorString(err) << "\n"; return -1; }

        err = cudaMemcpy(pDepth, input_depth.data(), w * h * sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) { std::cout << "Error copying memory to device: " << cudaGetErrorString(err) << "\n"; return -1; }

        {
            dim3 dimGrid(w / block_size, h / block_size);
            dim3 dimBlock(block_size, block_size);
            cudaEventRecord(evt[0]);
            gpu_blur_integral << <dimGrid, dimBlock >> > (pOutput, pInput, pDepth, w, h, focus_distance);
            err = cudaGetLastError();
            if (err != cudaSuccess) { std::cout << "CUDA error in kernel call: " << cudaGetErrorString(err) << "\n"; return -1; }
            cudaEventRecord(evt[1]);
        }

        err = cudaMemcpy(output.data(), pOutput, w * h * sizeof(color), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) { std::cout << "Error copying memory to host: " << cudaGetErrorString(err) << "\n"; return -1; }

        err = cudaFree(pInput);
        if (err != cudaSuccess) { std::cout << "Error freeing allocation: " << cudaGetErrorString(err) << "\n"; return -1; }

        err = cudaFree(pDepth);
        if (err != cudaSuccess) { std::cout << "Error freeing allocation: " << cudaGetErrorString(err) << "\n"; return -1; }

        err = cudaFree(pOutput);
        if (err != cudaSuccess) { std::cout << "Error freeing allocation: " << cudaGetErrorString(err) << "\n"; return -1; }

        cudaEventSynchronize(evt[1]);
        cudaEventElapsedTime(&dt, evt[0], evt[1]);

        for (auto& e : evt) { cudaEventDestroy(e); }
    }

    std::cout << "GPU Computation 1 took: " << dt << " ms\n";

    auto convert_and_write = [w, h, ch](std::string const& filename, std::vector<color> const& data)
    {
        std::vector<rawcolor> tmp(w * h * ch);
        std::transform(data.cbegin(), data.cend(), tmp.begin(),
            [](color c) { return rawcolor{ (unsigned char)(c.r * 255.0f),
                                            (unsigned char)(c.g * 255.0f),
                                            (unsigned char)(c.b * 255.0f),
                                            (unsigned char)(c.a * 255.0f) }; });

        int res = stbi_write_jpg(filename.c_str(), w, h, ch, tmp.data(), 40);
        if (res == 0)
        {
            std::cout << "Error writing output to file " << filename << "\n";
        }
        else { std::cout << "Output written to file " << filename << "\n"; }
    };

    convert_and_write(output_filename1, output);

    return 0;
}
