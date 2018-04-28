
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

int sigmaC_max=200;
int sigmaC=1;
int sigmaD=1;

float *d_gaussKernel;

__device__ float w(int i,int j ,int k ,int l,float sigmaC,float sigmaD,uchar pxIJ, uchar pxKL)
{
	//float a = (powf(i - k, 2) + powf(j - l, 2)) / (2 * powf(sigmaD, 2));
	float b = (powf(pxIJ-pxKL, 2)) / (2 * powf(sigmaC, 2));

	float w = expf(-b);//expf(-a-b);
	return w;
}
__device__ float wv2(float sigmaC, uchar pxIJ, uchar pxKL)
{
	float b = (powf(pxIJ - pxKL, 2)) / (2 * powf(sigmaC, 2));
	float w = expf(-b);
	return w;
}
__global__ void BilateralFiltering(uchar *source, uchar *dest, int size,float sigmaC,float sigmaD,float *gaussKernel)
{
	int idx = blockIdx.x*blockDim.x * 3 + threadIdx.x * 3;
	//int row = blockIdx.x* blockDim.x * 3;
	int rowSize= blockDim.x * 3;
	//int col = threadIdx.x * 3;
	int kernel = size / 2;

	for (int color = 0; color < 3; color++)
	{
		float sum1 = 0;
		float sum2 = 0;
		for (int x = -kernel; x <= kernel; x++)
		{
			for (int y = -kernel; y <= kernel; y++)
			{
				//float W= w(
				//	col, //960 = 320
				//	row, //5760 = 3
				//	col + y, //960-2 = 318
				//	row + x, //5760-2 = 1
				//	sigmaC,
				//	sigmaD,
				//	source[idx + color], //6720
				//	source[idx + color + x * rowSize + y * 3]) //6720+x*1920-6 = 2874 2877 2880 2883 2886
				//	* gaussKernel[(x+kernel)*size+y+kernel]; 

				float W = wv2(sigmaC, source[idx + color], source[idx + color + x * rowSize + y * 3])
					* gaussKernel[(x + kernel)*size + y + kernel];

				sum1 += (float)source[idx + color + x * rowSize + y * 3] * W;

				sum2 += W;

			}
		}
		dest[idx + color] = (uchar)(sum1/sum2);
	}
}

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
	//float A = 1 / (2 * CV_PI*powf(sigma, 2));

	int distX = x / 2;
	int distY = y / 2;

	for (int i = 0; i < x; i++)
	{
		for (int j = 0; j < y; j++)
		{
			valami[i*x+j] =expf(-(powf(i-distX, 2) + powf(j-distY, 2)) / (2 * powf(sigmaD, 2)));//*A;
			//cout << valami[i*x+j] << "\t";
		}
		//cout << endl;
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

void on_trackbar(int, void*)
{
	float *valami = new float[5*5];

	int distX = 5 / 2;
	int distY = 5 / 2;

	for (int i = 0; i < 5; i++)
	{
		for (int j = 0; j < 5; j++)
		{
			valami[i*5 + j] = expf(-(powf(i - distX, 2) + powf(j - distY, 2)) / (2 * powf(sigmaD, 2)));//*A;
		}
	}
	cudaMemcpy(d_gaussKernel, valami, sizeof(float)*5*5, cudaMemcpyHostToDevice);

	delete valami;
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
	GaussKernelGenerator(kernelX, kernelY, 2);


	VideoCapture camera(0);
	Mat          frame;

	if (!camera.isOpened())
		return -1;
	 
	namedWindow("Source");
	//namedWindow("Gauss");
	namedWindow("Bilateral");

	createTrackbar("Color space","Bilateral", &sigmaC, sigmaC_max);
	createTrackbar("Coord space", "Bilateral", &sigmaD, sigmaC_max, on_trackbar);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	camera >> frame;	

	rows = frame.rows;
	cols = frame.cols;
	int a = frame.type();

	uchar *d_source, *d_bilateral;//, *d_gauss;
	Mat source(frame.size(), a, createPointers(rows*cols * 3, &d_source));
	//Mat gauss(frame.size(), a, createPointers(rows*cols * 3, &d_gauss));
	Mat bilateral(frame.size(), a, createPointers(rows*cols * 3, &d_bilateral));

	while (true)
	{		
		camera >> frame;
		frame.copyTo(source);

		cudaEventRecord(start);
		{

			//GaussFilter << <rows, cols >> > (d_source, d_gauss,d_gaussKernel,kernelX,kernelY);
			BilateralFiltering << <rows, cols >> > (d_source, d_bilateral, 5, sigmaC,sigmaD,d_gaussKernel);

			cudaThreadSynchronize();
		}
		cudaEventRecord(stop);		

		float ms = 0.0f;
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&ms, start, stop);
		cout << "Elapsed GPU time: " << ms << " milliseconds" << endl;

		/*cudaEventRecord(start);
		{
			bilateralFilter(source, bilateral, 5, 1, 1);
		}
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&ms, start, stop);
		cout << "Elapsed CPU time: " << ms << " milliseconds" << endl;*/

		imshow("Source", frame);
		//imshow("Gauss", gauss);
		imshow("Bilateral", bilateral);


		if (waitKey(1) == 27) break;
	}

	return 0;
}