
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace cv;
using namespace std;


int rows;
int cols;

float *d_gaussKernel;

__global__ void GaussFilter(uchar *source,uchar *dest, float *kernel,int X,int Y)
{
	int idx = blockIdx.x*blockDim.x*3 + threadIdx.x*3;

	int kernelX = X / 2;
	int kernelY = Y / 2;
	int row = blockDim.x*3;

	for (int color = 0; color < 3; color++)
	{
		float sum = 0;
		for (int x = -kernelX; x <= kernelX; x++)
		{
			for (int y = -kernelY; y <= kernelY; y++)
			{
				sum += (float)source[idx + color+x*row+y*3]*kernel[(x+kernelX)*X+y+kernelY];
			}
		}
		dest[idx + color] =(uchar) sum;
	}

}

void GaussKernelGenerator(int x, int y,float sigma)
{
	float *valami=new float[x*y];
	float A = 1 / (2 * CV_PI*powf(sigma, 2));

	int distX = x / 2;
	int distY = y / 2;

	for (int i = 0; i < x; i++)
	{
		for (int j = 0; j < y; j++)
		{
			valami[i*x+j] = A * expf(-(powf(i-distX, 2) + powf(j-distY, 2)) / (2 * powf(sigma, 2)));
			cout << valami[i*x+j] << "\t";
		}
		cout << endl;
	}
		
	cudaMalloc((void**)&d_gaussKernel, sizeof(float)*x*y);
	cudaMemcpy(d_gaussKernel, valami, sizeof(float)*x*y, cudaMemcpyHostToDevice);

	delete valami;
}

uchar* createPointers(uint bytes, uchar **devicePtr)
{
	uchar *ptr = NULL;
	cudaSetDeviceFlags(cudaDeviceMapHost);
	cudaHostAlloc(&ptr, bytes, cudaHostAllocMapped);
	cudaHostGetDevicePointer(devicePtr, ptr, 0);
	return ptr;
}

int main(int argc, char** argv)
{
	/*cudaDeviceProp  prop;
	int whichDevice;
	cudaGetDevice(&whichDevice);
	cudaGetDeviceProperties(&prop, whichDevice);
	cout << prop.name << "\t" << prop.canMapHostMemory;*/

	int kernelX = 5;
	int kernelY = 5;
	GaussKernelGenerator(kernelX, kernelY, 3);


	VideoCapture camera(0);
	Mat          frame;

	if (!camera.isOpened())
		return -1;
	 
	namedWindow("Source");
	namedWindow("Gauss");
	//namedWindow("Bilateral");

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	camera >> frame;	

	rows = frame.rows;
	cols = frame.cols;
	int a = frame.type();

	uchar *d_source, *d_gauss, *d_bilateral;
	Mat source(frame.size(), a, createPointers(rows*cols * 3, &d_source));
	Mat gauss(frame.size(), a, createPointers(rows*cols * 3, &d_gauss));
	//Mat bilateral(frame.size(), a, createPointers(rows*cols * 3, &d_bilateral));

	while (true)
	{		
		camera >> frame;
		frame.copyTo(source);

		cudaEventRecord(start);
		{

			GaussFilter << <rows, cols >> > (d_source, d_gauss,d_gaussKernel,kernelX,kernelY);
			//GaussFilter << <rows, cols >> > (d_gauss, d_bilateral, d_gaussKernel, kernelX, kernelY);

			cudaThreadSynchronize();
		}
		cudaEventRecord(stop);		

		float ms = 0.0f;
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&ms, start, stop);
		cout << "Elapsed GPU time: " << ms << " milliseconds" << endl;


		imshow("Source", frame);
		imshow("Gauss", gauss);
		//imshow("Bilateral", bilateral);


		if (waitKey(1) == 27) break;
	}

	return 0;
}