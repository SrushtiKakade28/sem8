%%writefile bubble.cu
#include <iostream>
#include <vector>
#include <chrono>
using namespace std;
__device__ void device_swap(int& a, int& b) {
int temp = a;
a = b;
b = temp;
}
__global__ void kernel_bubble_sort_odd_even(int* arr, int size) {
bool isSorted = false;
while (!isSorted) {
isSorted = true;
int tid = blockIdx.x * blockDim.x + threadIdx.x; //calculating
gloable thread id.
if (tid % 2 == 0 && tid < size - 1) {
if (arr[tid] > arr[tid + 1]) {
device_swap(arr[tid], arr[tid + 1]);
isSorted = false;
}
}
__syncthreads(); // Synchronize threads within block
if (tid % 2 != 0 && tid < size - 1) {
if (arr[tid] > arr[tid + 1]) {
device_swap(arr[tid], arr[tid + 1]);
isSorted = false;
}
}
__syncthreads(); // Synchronize threads within block
}
}
void bubble_sort_odd_even(vector<int>& arr) {
int size = arr.size();
int* d_arr;
cudaMalloc(&d_arr, size * sizeof(int));
cudaMemcpy(d_arr, arr.data(), size * sizeof(int),
cudaMemcpyHostToDevice);
// Calculate grid and block dimensions
int blockSize = 256;
int gridSize = (size + blockSize - 1) / blockSize;
// Perform bubble sort on GPU
kernel_bubble_sort_odd_even<<<gridSize, blockSize>>>(d_arr, size);
// Copy sorted array back to host
cudaMemcpy(arr.data(), d_arr, size * sizeof(int),
cudaMemcpyDeviceToHost);
cout<<"sorted array"<<endl;
for(int i=0;i<size;i++){
cout<<arr[i]<<" ";
}
cout<<endl;
cudaFree(d_arr);
}
int main() {
vector<int> arr = {5,4 , 3,2 ,1 ,0,6,9,7 };
double start, end;
// Measure performance of parallel bubble sort using odd-even
transposition
start =
chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now().time
_since_epoch()).count();
bubble_sort_odd_even(arr);
end =
chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now().time
_since_epoch()).count();
cout << "Parallel bubble sort using odd-even transposition time: " <<
end - start << " milliseconds" << endl;
return 0;
}
!nvcc bubble.cu -o bubble
!./bubble
OUTPUT :
sorted array
0 1 2 3 4 5 6 7 9
Parallel bubble sort using odd-even transposition time: 101 milliseconds
CODE 2 - MERGE SORT :
%%writefile merge_sort.cu
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm> // for min function
using namespace std;
// Kernel to merge two sorted halves
__global__ void kernel_merge(int* arr, int* temp, int* subarray_sizes, int
array_size) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;//calculating global
thread id
int left_start = idx * 2 * (*subarray_sizes);
if (left_start < array_size) {
int mid = min(left_start + (*subarray_sizes) - 1, array_size - 1);
int right_end = min(left_start + 2 * (*subarray_sizes) - 1,
array_size - 1);
int i = left_start;
int j = mid + 1;
int k = left_start;
// Merge process
while (i <= mid && j <= right_end) {
if (arr[i] <= arr[j]) {
temp[k] = arr[i];
i++;
} else {
temp[k] = arr[j];
j++;
}
k++;
}
while (i <= mid) {
temp[k] = arr[i];
i++;
k++;
}
while (j <= right_end) {
temp[k] = arr[j];
j++;
k++;
}
// Copy the sorted subarray back to the original array
for (int t = left_start; t <= right_end; t++) {
arr[t] = temp[t];
}
}
}
void merge_sort(vector<int>& arr) {
int array_size = arr.size();
int* d_arr;
int* d_temp;
int* d_subarray_size;
// Allocate memory on the GPU
cudaMalloc(&d_arr, array_size * sizeof(int));
cudaMalloc(&d_temp, array_size * sizeof(int));
cudaMalloc(&d_subarray_size, sizeof(int)); // Holds the subarray size
for each step
cudaMemcpy(d_arr, arr.data(), array_size * sizeof(int),
cudaMemcpyHostToDevice);
int blockSize = 256; // Threads per block
int gridSize; // Number of blocks in the grid, depending on the
subarray size
// Start with width of 1, then double each iteration
int width = 1;
while (width < array_size) {
cudaMemcpy(d_subarray_size, &width, sizeof(int),
cudaMemcpyHostToDevice);
gridSize = (array_size / (2 * width)) + 1;
kernel_merge<<<gridSize, blockSize>>>(d_arr, d_temp,
d_subarray_size, array_size);
cudaDeviceSynchronize(); // Ensure all threads finish before the
next step
// Double the subarray width for the next iteration
width *= 2;
}
// Copy the sorted array back to the host
cudaMemcpy(arr.data(), d_arr, array_size * sizeof(int),
cudaMemcpyDeviceToHost);
// Free GPU memory
cudaFree(d_arr);
cudaFree(d_temp);
cudaFree(d_subarray_size);
}
int main() {
vector<int> arr = {6, 5, 4, 1, 7, 9, 8, 3, 2};
double start, end;
start =
chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now().time
_since_epoch()).count();
merge_sort(arr);
end =
chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now().time
_since_epoch()).count();
cout << "Parallel merge sort time: " << end - start << " milliseconds"
<< endl;
cout << "Sorted array: ";
for (int num : arr) {
cout << num << " ";
}
cout << endl;
return 0;
}
!nvcc merge_sort.cu -o merge
!./merge
OUTPUT :
Parallel merge sort time: 199 milliseconds
Sorted array: 1 2 3 4 5 6 7 8 9
