%%writefile add.cu
#include <iostream>
#include <cstdlib> // Include <cstdlib> for rand()
using namespace std;
__global__
void add(int* A, int* B, int* C, int size) {
int tid = blockIdx.x * blockDim.x + threadIdx.x;
if (tid < size) {
C[tid] = A[tid] + B[tid];
}
}
void print(int* vector, int size) {
for (int i = 0; i < size; i++) {
cout << vector[i] << " ";
}
cout << endl;
}
int main() {
int N;
cout << "Enter the size of the vectors: ";
cin >> N;
int* A, * B, * C;
int vectorSize = N;
size_t vectorBytes = vectorSize * sizeof(int);
// Allocate host memory
A = new int[vectorSize];
B = new int[vectorSize];
C = new int[vectorSize];
// Initialize host arrays
cout << "Enter elements of vector A:" << endl;
for (int i = 0; i < N; i++) {
cin >> A[i];
}
cout << "Enter elements of vector B:" << endl;
for (int i = 0; i < N; i++) {
cin >> B[i];
}
cout << "Vector A: ";
print(A, N);
cout << "Vector B: ";
print(B, N);
int* X, * Y, * Z;
// Allocate device memory
cudaMalloc(&X, vectorBytes);
cudaMalloc(&Y, vectorBytes);
cudaMalloc(&Z, vectorBytes);
// Check for CUDA memory allocation errors
if (X == nullptr || Y == nullptr || Z == nullptr) {
cerr << "CUDA memory allocation failed" << endl;
return 1;
}
// Copy data from host to device
cudaMemcpy(X, A, vectorBytes, cudaMemcpyHostToDevice);
cudaMemcpy(Y, B, vectorBytes, cudaMemcpyHostToDevice);
int threadsPerBlock = 256;
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
// Launch kernel
add<<<blocksPerGrid, threadsPerBlock>>>(X, Y, Z, N);
// Check for kernel launch errors
cudaError_t kernelLaunchError = cudaGetLastError();
if (kernelLaunchError != cudaSuccess) {
cerr << "CUDA kernel launch failed: " <<
cudaGetErrorString(kernelLaunchError) << endl;
return 1;
}
// Copy result from device to host
cudaMemcpy(C, Z, vectorBytes, cudaMemcpyDeviceToHost);
// Check for CUDA memcpy errors
cudaError_t memcpyError = cudaGetLastError();
if (memcpyError != cudaSuccess) {
cerr << "CUDA memcpy failed: " << cudaGetErrorString(memcpyError) <<
endl;
return 1;
}
cout << "Addition: ";
print(C, N);
// Free device memory
cudaFree(X);
cudaFree(Y);
cudaFree(Z);
// Free host memory
delete[] A;
delete[] B;
delete[] C;
return 0;
}
!nvcc add.cu -o add
!./add
OUTPUT :
Enter the size of the vectors: 3
Enter elements of vector A:
1 2 3
Enter elements of vector B:
4 5 6
Vector A: 1 2 3
Vector B: 4 5 6
Addition: 5 7 9
