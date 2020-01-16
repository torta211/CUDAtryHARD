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

__global__ void gpu_blur_integral(float4* render_out, float4* integral_in, int W, int H)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    // radius of kernal will be = 3 pixels, this is just a try
    // corner1, corner2
    // corner3, corner4
    int corner1_x = x - 3 > 0 ? x - 3 : 0;
    int corner1_y = y - 3 > 0 ? y - 3 : 0;
    float4 corner1 = integral_in[corner1_y * W + corner1_x];

    int corner2_x = x + 3 < W ? x + 3 : W - 1;
    int corner2_y = y - 3 > 0 ? y - 3 : 0;
    float4 corner2 = integral_in[corner2_y * W + corner2_x];

    int corner3_x = x - 3 > 0 ? x - 3 : 0;
    int corner3_y = y + 3 < H ? y + 3 : H - 1;
    float4 corner3 = integral_in[corner3_y * W + corner3_x];

    int corner4_x = x + 3 < W ? x + 3 : W - 1;
    int corner4_y = y + 3 < H ? y + 3 : H - 1;
    float4 corner4 = integral_in[corner4_y * W + corner4_x];

    float blurred_val_r = corner4.x - corner3.x - corner2.x + corner1.x;
    float blurred_val_g = corner4.y - corner3.y - corner2.y + corner1.y;
    float blurred_val_b = corner4.z - corner3.z - corner2.z + corner1.z;
    float blurred_val_a = corner4.w - corner3.w - corner2.w + corner1.w;
    blurred_val_r /= (corner2_x - corner1_x + 1) * (corner3_y - corner1_y + 1);
    blurred_val_g /= (corner2_x - corner1_x + 1) * (corner3_y - corner1_y + 1);
    blurred_val_b /= (corner2_x - corner1_x + 1) * (corner3_y - corner1_y + 1);
    blurred_val_a /= (corner2_x - corner1_x + 1) * (corner3_y - corner1_y + 1);

    render_out[y * W + x] = float4{ blurred_val_r, blurred_val_g, blurred_val_b, blurred_val_a};
}

__global__ void gpu_rotate_buffer(float4* output, float4* input, int W, int H, float angle)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    float u = x - W / 2.0f;
    float v = y - H / 2.0f;

    float u0 = u * std::cos(angle) - v * std::sin(angle) + W / 2.0f;
    float v0 = v * std::cos(angle) + u * std::sin(angle) + H / 2.0f;

    int ui0 = (int)u0;
    int ui1 = (int)u0 + 1;
    int vi0 = (int)v0;
    int vi1 = (int)v0 + 1;

    float4 c0 = (ui0 >= 0 && vi0 >= 0 && vi0 < H && ui0 < W) ? input[vi0 * W + ui0] : float4{ 0, 0, 0, 0 };
    float4 c1 = (ui0 >= 0 && vi1 >= 0 && vi1 < H && ui0 < W) ? input[vi1 * W + ui0] : float4{ 0, 0, 0, 0 };
    float4 c2 = (ui1 >= 0 && vi0 >= 0 && vi0 < H && ui1 < W) ? input[vi0 * W + ui1] : float4{ 0, 0, 0, 0 };
    float4 c3 = (ui1 >= 0 && vi1 >= 0 && vi1 < H && ui1 < W) ? input[vi1 * W + ui1] : float4{ 0, 0, 0, 0 };

    //bilinear interpolation:
    float ufrac = ui0 + 1 - u0;
    float vfrac = vi0 + 1 - v0;

    float Ar = c0.x * ufrac + c2.x * (1 - ufrac);
    float Ag = c0.y * ufrac + c2.y * (1 - ufrac);
    float Ab = c0.z * ufrac + c2.z * (1 - ufrac);
    float Aa = c0.w * ufrac + c2.w * (1 - ufrac);

    float Br = c1.x * ufrac + c3.x * (1 - ufrac);
    float Bg = c1.y * ufrac + c3.y * (1 - ufrac);
    float Bb = c1.z * ufrac + c3.z * (1 - ufrac);
    float Ba = c1.w * ufrac + c3.w * (1 - ufrac);

    float Cr = Ar * vfrac + Br * (1 - vfrac);
    float Cg = Ag * vfrac + Bg * (1 - vfrac);
    float Cb = Ab * vfrac + Bb * (1 - vfrac);
    float Ca = Aa * vfrac + Ba * (1 - vfrac);

    Cr = Cr < 0 ? 0 : Cr;
    Cg = Cg < 0 ? 0 : Cg;
    Cb = Cb < 0 ? 0 : Cb;
    Ca = Ca < 0 ? 0 : Ca;

    output[y * W + x] = float4{ Cr, Cg, Cb, Ca };
}

__global__ void gpu_rotate_texture(float4* output, cudaTextureObject_t input, int W, int H, float angle)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    float u = x - W / 2.0f;
    float v = y - H / 2.0f;

    float u0 = u * std::cos(angle) - v * std::sin(angle) + W / 2.0f + 0.5f;
    float v0 = v * std::cos(angle) + u * std::sin(angle) + H / 2.0f + 0.5f;

    // Read from texture and write to global memory
    output[y * W + x] = tex2D<float4>(input, u0, v0);
}

int main()
{
    static const std::string input_filename = "3dindoorrender.png";
    static const std::string output_filename1 = "gpu_out1.jpg";
    static const std::string output_filename2 = "gpu_out2.jpg";

    static const int block_size = 32;

    std::cout << "Enter rotation angle in degrees:\n";
    float angle = 0.5f;
    std::cin >> angle;
    angle *= 3.1415926535f / 180.0f;

    int w = 0;//width
    int h = 0;//height
    int ch = 0;//number of components
    
    rawcolor* data0 = reinterpret_cast<rawcolor*>(stbi_load(input_filename.c_str(), &w, &h, &ch, 4 /* we expect 4 components */));
    if (!data0)
    {
        std::cout << "Error: could not open input file: " << input_filename << "\n";
        return -1;
    }
    else
    {
        std::cout << "Image (" << input_filename << ") opened successfully. Width x Height x Components = " << w << " x " << h << " x " << ch << "\n";
    }

    std::vector<color> input(w * h);
    std::vector<color> output1(w * h);
    std::vector<color> output2(w * h);

    std::transform(data0, data0 + w * h, input.begin(), [](rawcolor c) { return color{ c.r / 255.0f, c.g / 255.0f, c.b / 255.0f, c.a / 255.0f }; });
    stbi_image_free(data0);

    std::vector<color> input_integral = cpu_create_integral(input, w, h);

    //GPU version using buffers:
    float dt = 0.0f;//milliseconds
    {
        float4* pInput = nullptr;
        float4* pOutput = nullptr;

        cudaEvent_t evt[2];
        for (auto& e : evt) { cudaEventCreate(&e); }

        cudaError_t err = cudaSuccess;
        err = cudaMalloc((void**)&pInput, w * h * sizeof(color));
        if (err != cudaSuccess) { std::cout << "Error allocating CUDA memory: " << cudaGetErrorString(err) << "\n"; return -1; }

        err = cudaMalloc((void**)&pOutput, w * h * sizeof(color));
        if (err != cudaSuccess) { std::cout << "Error allocating CUDA memory: " << cudaGetErrorString(err) << "\n"; return -1; }

        err = cudaMemcpy(pInput, input_integral.data(), w * h * sizeof(color), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) { std::cout << "Error copying memory to device: " << cudaGetErrorString(err) << "\n"; return -1; }

        {
            dim3 dimGrid(w / block_size, h / block_size);
            dim3 dimBlock(block_size, block_size);
            cudaEventRecord(evt[0]);
            gpu_blur_integral << <dimGrid, dimBlock >> > (pOutput, pInput, w, h);
            err = cudaGetLastError();
            if (err != cudaSuccess) { std::cout << "CUDA error in kernel call: " << cudaGetErrorString(err) << "\n"; return -1; }
            cudaEventRecord(evt[1]);
        }

        err = cudaMemcpy(output1.data(), pOutput, w * h * sizeof(color), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) { std::cout << "Error copying memory to host: " << cudaGetErrorString(err) << "\n"; return -1; }

        err = cudaFree(pInput);
        if (err != cudaSuccess) { std::cout << "Error freeing allocation: " << cudaGetErrorString(err) << "\n"; return -1; }

        err = cudaFree(pOutput);
        if (err != cudaSuccess) { std::cout << "Error freeing allocation: " << cudaGetErrorString(err) << "\n"; return -1; }

        cudaEventSynchronize(evt[1]);
        cudaEventElapsedTime(&dt, evt[0], evt[1]);

        for (auto& e : evt) { cudaEventDestroy(e); }
    }

    float dt2 = 0.0f;//milliseconds
    {
        cudaError_t err = cudaSuccess;

        cudaEvent_t evt[2];
        for (auto& e : evt) { cudaEventCreate(&e); }

        //Channel layout of data:
        cudaChannelFormatDesc channelDescInput = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);

        //Allocate data:
        cudaArray* aInput;

        err = cudaMallocArray(&aInput, &channelDescInput, w, h);
        if (err != cudaSuccess) { std::cout << "Error allocating CUDA memory: " << cudaGetErrorString(err) << "\n"; return -1; }

        //Upload data to device:
        err = cudaMemcpyToArray(aInput, 0, 0, input.data(), w * h * sizeof(color), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) { std::cout << "Error copying memory to device: " << cudaGetErrorString(err) << "\n"; return -1; }

        //Specify texture resource description:
        cudaResourceDesc resdescInput{};
        resdescInput.resType = cudaResourceTypeArray;
        resdescInput.res.array.array = aInput;

        //Specify texture description:
        cudaTextureDesc texDesc{};
        texDesc.addressMode[0] = cudaAddressModeBorder;
        texDesc.addressMode[1] = cudaAddressModeBorder;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        // Create texture object
        cudaTextureObject_t texObjInput = 0;
        err = cudaCreateTextureObject(&texObjInput, &resdescInput, &texDesc, nullptr);
        if (err != cudaSuccess) { std::cout << "Error creating texture object: " << cudaGetErrorString(err) << "\n"; return -1; }

        //The output is just an usual buffer:
        float4* pOutput = nullptr;
        err = cudaMalloc((void**)&pOutput, w * h * sizeof(color));
        if (err != cudaSuccess) { std::cout << "Error allocating CUDA memory: " << cudaGetErrorString(err) << "\n"; return -1; }

        // Invoke kernel
        {
            dim3 dimGrid(w / block_size, h / block_size);
            dim3 dimBlock(block_size, block_size);
            cudaEventRecord(evt[0]);
            gpu_rotate_texture << <dimGrid, dimBlock >> > (pOutput, texObjInput, w, h, angle);
            err = cudaGetLastError();
            if (err != cudaSuccess) { std::cout << "CUDA error in kernel call: " << cudaGetErrorString(err) << "\n"; return -1; }
            cudaEventRecord(evt[1]);
        }

        err = cudaMemcpy(output2.data(), pOutput, w * h * sizeof(color), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) { std::cout << "Error copying memory to host: " << cudaGetErrorString(err) << "\n"; return -1; }

        //Cleanup:
        err = cudaDestroyTextureObject(texObjInput);
        if (err != cudaSuccess) { std::cout << "Error destroying texture object: " << cudaGetErrorString(err) << "\n"; return -1; }

        err = cudaFreeArray(aInput);
        if (err != cudaSuccess) { std::cout << "Error freeing array allocation: " << cudaGetErrorString(err) << "\n"; return -1; }

        err = cudaFree(pOutput);
        if (err != cudaSuccess) { std::cout << "Error freeing allocation: " << cudaGetErrorString(err) << "\n"; return -1; }

        cudaEventSynchronize(evt[1]);
        cudaEventElapsedTime(&dt2, evt[0], evt[1]);

        for (auto& e : evt) { cudaEventDestroy(e); }
    }

    std::cout << "GPU Computation 1 took: " << dt << " ms\n";
    std::cout << "GPU Computation 2 took: " << dt2 << " ms\n";

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

    convert_and_write(output_filename1, output1);
    convert_and_write(output_filename2, output2);

    return 0;
}
