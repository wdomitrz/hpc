#include "common/cpu_bitmap.h"
#include "common/errors.h"
#include "cuda.h"

#define DIM 1024
#define rnd(x) (x * rand() / RAND_MAX)
#define INF 2e10f
#define SPHERES 100

__constant__ float spheres_x[SPHERES], spheres_y[SPHERES], spheres_z[SPHERES],
    spheres_red[SPHERES], spheres_green[SPHERES], spheres_blue[SPHERES],
    spheres_radius[SPHERES];
float hostSpheres_x[SPHERES], hostSpheres_y[SPHERES], hostSpheres_z[SPHERES],
    hostSpheres_red[SPHERES], hostSpheres_green[SPHERES],
    hostSpheres_blue[SPHERES], hostSpheres_radius[SPHERES];

__device__ float hit(int i, float bitmapX, float bitmapY, float *colorFalloff) {
    float distX = bitmapX - spheres_x[i];
    float distY = bitmapY - spheres_y[i];

    if (distX * distX + distY * distY < spheres_radius[i] * spheres_radius[i]) {
        float distZ = sqrtf(spheres_radius[i] * spheres_radius[i] -
                            distX * distX - distY * distY);
        *colorFalloff = distZ / spheres_radius[i];
        return distZ + spheres_z[i];
    }

    return -INF;
}

__global__ void kernel(unsigned char *bitmap) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float bitmapX = (x - DIM / 2);
    float bitmapY = (y - DIM / 2);

    float red = 0, green = 0, blue = 0;
    float maxDepth = -INF;

    for (int i = 0; i < SPHERES; i++) {
        float colorFalloff;
        float depth = hit(i, bitmapX, bitmapY, &colorFalloff);

        if (depth > maxDepth) {
            red = spheres_red[i] * colorFalloff;
            green = spheres_green[i] * colorFalloff;
            blue = spheres_blue[i] * colorFalloff;
            maxDepth = depth;
        }
    }

    bitmap[offset * 4 + 0] = (int)(red * 255);
    bitmap[offset * 4 + 1] = (int)(green * 255);
    bitmap[offset * 4 + 2] = (int)(blue * 255);
    bitmap[offset * 4 + 3] = 255;
}

struct DataBlock {
    unsigned char *hostBitmap;
    // Sphere *spheres;
};

int main(void) {
    DataBlock data;
    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start, 0));

    CPUBitmap bitmap(DIM, DIM, &data);
    unsigned char *devBitmap;
    // Sphere *devSpheres;

    HANDLE_ERROR(cudaMalloc((void **)&devBitmap, bitmap.image_size()));
    // HANDLE_ERROR(cudaMalloc((void **)&devSpheres, sizeof(Sphere) * SPHERES));

    // Sphere *hostSpheres = (Sphere *)malloc(sizeof(Sphere) * SPHERES);

    for (int i = 0; i < SPHERES; i++) {
        hostSpheres_red[i] = rnd(1.0f);
        hostSpheres_green[i] = rnd(1.0f);
        hostSpheres_blue[i] = rnd(1.0f);
        hostSpheres_x[i] = rnd(1000.0f) - 500;
        hostSpheres_y[i] = rnd(1000.0f) - 500;
        hostSpheres_z[i] = rnd(1000.0f) - 500;
        hostSpheres_radius[i] = rnd(100.0f) + 20;
    }

    HANDLE_ERROR(
        cudaMemcpyToSymbol(spheres_x, hostSpheres_x, sizeof(float) * SPHERES));
    HANDLE_ERROR(
        cudaMemcpyToSymbol(spheres_y, hostSpheres_y, sizeof(float) * SPHERES));
    HANDLE_ERROR(
        cudaMemcpyToSymbol(spheres_z, hostSpheres_z, sizeof(float) * SPHERES));
    HANDLE_ERROR(cudaMemcpyToSymbol(spheres_radius, hostSpheres_radius,
                                    sizeof(float) * SPHERES));
    HANDLE_ERROR(cudaMemcpyToSymbol(spheres_blue, hostSpheres_blue,
                                    sizeof(float) * SPHERES));
    HANDLE_ERROR(cudaMemcpyToSymbol(spheres_green, hostSpheres_green,
                                    sizeof(float) * SPHERES));
    HANDLE_ERROR(cudaMemcpyToSymbol(spheres_red, hostSpheres_red,
                                    sizeof(float) * SPHERES));

    dim3 grids(DIM / 16, DIM / 16);
    dim3 threads(16, 16);
    kernel<<<grids, threads>>>(devBitmap);

    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));

    float elapsedTime;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("Time to generate: %3.1f ms\n", elapsedTime);

    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));

    HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), devBitmap, bitmap.image_size(),
                            cudaMemcpyDeviceToHost));

    bitmap.dump_ppm("image.ppm");

    cudaFree(devBitmap);
}
