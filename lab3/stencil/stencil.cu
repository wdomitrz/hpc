#include <stdio.h>
#include <time.h>

#define RADIUS 100
#define NUM_ELEMENTS 500000

static void handleError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
#define cudaCheck(err) (handleError(err, __FILE__, __LINE__))

__global__ void stencil_1d(int *in, int *out) {
    int i = threadIdx.x + blockIdx.y * gridDim.x;

    if (i < NUM_ELEMENTS) {
        out[i] = 0;
        for (int j = i - RADIUS; j <= i + RADIUS; j++)
            if (0 <= j && j < NUM_ELEMENTS) out[i] += out[j];
    }
}

void cpu_stencil_1d(int *in, int *out) {
    for (int i = 0; i < NUM_ELEMENTS; i++) {
        out[i] = 0;
        for (int j = i - RADIUS; j <= i + RADIUS; j++)
            if (0 <= j && j < NUM_ELEMENTS) out[i] += out[j];
    }
}

int main() {
    // PUT YOUR CODE HERE - INPUT AND OUTPUT ARRAYS
    int in[NUM_ELEMENTS], out[NUM_ELEMENTS], out2[NUM_ELEMENTS];
    int *inGPU, *outGPU;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // PUT YOUR CODE HERE - DEVICE MEMORY ALLOCATION
    cudaMalloc((void **)&inGPU, NUM_ELEMENTS * sizeof(int));
    cudaMalloc((void **)&outGPU, NUM_ELEMENTS * sizeof(int));

    // Memory initialization
    cudaMemcpy(inGPU, in, NUM_ELEMENTS * sizeof(int), cudaMemcpyHostToDevice);

    // PUT YOUR CODE HERE - KERNEL EXECUTION
    stencil_1d<<<NUM_ELEMENTS / 256 + 1, 256>>>(inGPU, outGPU);

    cudaCheck(cudaPeekAtLastError());

    // PUT YOUR CODE HERE - COPY RESULT FROM DEVICE TO HOST
    cudaMemcpy(out2, outGPU, NUM_ELEMENTS * sizeof(int),
               cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Total GPU execution time:  %3.1f ms\n", elapsedTime);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // PUT YOUR CODE HERE - FREE DEVICE MEMORY
    cudaFree(inGPU);
    cudaFree(outGPU);

    struct timespec cpu_start, cpu_stop;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu_start);

    cpu_stencil_1d(in, out);

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu_stop);
    double result = (cpu_stop.tv_sec - cpu_start.tv_sec) * 1e3 +
                    (cpu_stop.tv_nsec - cpu_start.tv_nsec) / 1e6;
    printf("CPU execution time:  %3.1f ms\n", result);

    // Verify
    for (int i = 0; i < NUM_ELEMENTS; i++) {
        if (out[i] != out2[i]) {
            printf("ERROR\n");
            return 1;
        }
    }
    printf("OK\n");
    return 0;
}
