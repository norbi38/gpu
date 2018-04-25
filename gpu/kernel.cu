
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>



__device__ float gauss(float x[], float omega)
{
	float pikob = pow(2 * CV_PI, 3);
	
	float a = 1 / (pow(omega, 3)*sqrt(pikob));

	float gauss = a * exp(-(pow(x[0], 2) + pow(x[1], 2) + pow(x[2], 2)) / pow(omega, 2));

	return gauss;
}
__global__ void gaussFilter(cv::Mat dev_Image)
{
	float x = blockIdx.x;
	float y = threadIdx.x;
	float in = dev_Image.at<float>(x, y);
	
	dev_Image.at<float>(x, y)= gauss(&in, 3);
}


int main(int argc, char** argv)
{

	// Open a webcamera
	cv::VideoCapture camera(0);
	cv::Mat          frame;
	cv::Mat          source;
	if (!camera.isOpened())
		return -1;
	 
	// Create the capture windows
	cv::namedWindow("Source");
	cv::namedWindow("Gauss");

	// Create the cuda event timers 
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);


	// Create CPU/GPU shared images - one for the initial and one for the result
	camera >> source;
	frame = source;

	/*cv::Mat dev_Image=NULL;
	cudaMalloc(&dev_Image, sizeof(frame));*/
	while (1)
	{
		// Capture the image and store a gray conversion to the gpu
		camera >> source;
		frame = source;
		// Record the time it takes to process
		cudaEventRecord(start);
		{

			//for (size_t i = 0; i < frame.rows; i++)
			//{
			//	for (size_t j = 0; j < frame.cols; j++)
			//	{
			//		////cv::Vec3b a = frame.at<cv::Vec3b>(cv::Point(i, j));
			//		//std::cout << "B: " << (int)a[0] << " G: " << (int)a[1] << " C: " << (int)a[2] << std::endl;					
			//		/*a[0] =(int)a[0] / 10;
			//		a[1] =(int)a[1] / 10;
			//		a[2] =(int)a[2] / 10;*/
			//		frame.at<uchar>(i, j) /= 10;
			//		
			//	}				
			//}
			int width = frame.size().width;
			int height = frame.size().height;

			gaussFilter << <height, width >> > (&frame);

			cudaThreadSynchronize();

			//cudaMemcpyFromSymbol(&frame, &dev_Image, sizeof(dev_Image));
		}
		cudaEventRecord(stop);

		//Display the elapsed time
		float ms = 0.0f;
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&ms, start, stop);
		std::cout << "Elapsed GPU time: " << ms << " milliseconds" << std::endl;

		// Show the results
		cv::imshow("Source", source);
		cv::imshow("Gauss", frame);

		// Spin
		if (cv::waitKey(1) == 27) break;
	}

	return 0;
}