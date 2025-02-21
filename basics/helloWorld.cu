#include<cuda_runtime.h>
#include<device_launch_parameters.h>

#include<iostream>

__global__ void sayHelloWorld();

int main() {
    printf("Hello World! CPU \n");
    sayHelloWorld<<<1, 10>>>();

    cudaDeviceReset();

    system("pause");

    return 0;
}

__global__ void sayHelloWorld() {
    printf("Hello World! GPU \n");
}
