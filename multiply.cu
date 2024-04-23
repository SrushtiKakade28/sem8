%%writefile matrix_mult.cu
#include <iostream>
#include <cuda.h>
using namespace std;
#define BLOCK_SIZE 1
__global__ void gpuMM(float *A, float *B, float *C, int N) {
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
float sum = 0.f;
for (int n = 0; n < N; ++n)
sum += A[row * N + n] * B[n * N + col];
C[row * N + col] = sum;
}
int main(int argc, char *argv[]) {
int N;
// Get matrix size from user
cout << "Enter size of matrix (N): ";
cin >> N;
if (N % BLOCK_SIZE != 0) {
cerr << "Matrix size must be a multiple of BLOCK_SIZE." << endl;
return 1;
}
cout << "\nExecuting Matrix Multiplication" << endl;
cout << "Matrix size: " << N << "x" << N << endl;
// Allocate memory for matrices on the host
float *hA, *hB, *hC;
hA = new float[N * N];
hB = new float[N * N];
hC = new float[N * N];
// Read matrices from user
cout << "Enter elements of matrix A (" << N << "x" << N << "):" << endl;
for (int i = 0; i < N * N; ++i)
cin >> hA[i];
cout << "Enter elements of matrix B (" << N << "x" << N << "):" << endl;
for (int i = 0; i < N * N; ++i)
cin >> hB[i];
// Allocate memory for matrices on the device
int size = N * N * sizeof(float);
float *dA, *dB, *dC;
cudaMalloc(&dA, size);
cudaMalloc(&dB, size);
cudaMalloc(&dC, size);
// Copy matrices from the host to the device
cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice);
cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice);
dim3 threadBlock(BLOCK_SIZE, BLOCK_SIZE);
dim3 grid(N / BLOCK_SIZE, N / BLOCK_SIZE);
// Execute the matrix multiplication kernel
gpuMM<<<grid, threadBlock>>>(dA, dB, dC, N);
// Copy the result matrix from the device to the host
cudaMemcpy(hC, dC, size, cudaMemcpyDeviceToHost);
// Display the result matrix
cout << "\nResultant matrix:\n";
for (int row = 0; row < N; row++) {
for (int col = 0; col < N; col++) {
cout << hC[row * N + col] << " ";
}
cout << endl;
}
// Free device memory
cudaFree(dA);
cudaFree(dB);
cudaFree(dC);
// Free host memory
delete[] hA;
delete[] hB;
delete[] hC;
cout << "Finished." << endl;
return 0;
}
!nvcc matrix_mult.cu -o matrix_mult
!./matrix_mult
OUTPUT :
Enter size of matrix (N): 3
Executing Matrix Multiplication
Matrix size: 3x3
Enter elements of matrix A (3x3):
1 2 3
4 5 6
7 8 9
Enter elements of matrix B (3x3):
9 8 7
6 5 4
3 2 1
Resultant matrix:
30 24 18
84 69 54
138 114 90
Finished.
