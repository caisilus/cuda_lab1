#include <iostream>
#include <format>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include "lab.h"

using namespace std;
using namespace cv;

const int BLK = 100;
const int TPB = 512;

const RGB src_color{ 0, 0, 255 };
const RGB dest_color{ 0, 0, 0 };
float threshold = 128;


int main(int argc, char** argv)
{
	if (argc < 8)
	{
		cerr << std::format("Usage: {} input.img output-cpu.img output-gpu.img x1 y1 x2 y2\n"
			"  Where\n"
			"    x1, y1 are coordinates of upper left point of selection area\n"
			"    x2, y2 - lower right point of the same area\n", argv[0]);
		return -1;
	}
#pragma region Init
	Mat img = imread(argv[1], IMREAD_COLOR);
	//cout << img.rows << "x" << img.cols << endl;
	if (img.empty())
	{
		cerr << "Ошибка: не удалось загрузить изображение" << endl;
		return -1;
	}
	Coord ul{ atoi(argv[4]), atoi(argv[5]) };
	Coord lr{ atoi(argv[6]), atoi(argv[7]) };

	// cout << std::format("{} {} {} {}", ul.x, ul.y, lr.x, lr.y) << endl;
	
	Mat img_cpu = img.clone();
	Mat img_gpu = img.clone();

	cudaEvent_t startCUDA, stopCUDA;
	clock_t startCPU, stopCPU;
	float elapsedTimeCUDA, elapsedTimeCPU;
	cudaEventCreate(&startCUDA);
	cudaEventCreate(&stopCUDA);

	unsigned char* device_img;
	int N = img.rows * img.cols * img.channels();
#pragma endregion

#pragma region CPU
	startCPU = clock();
	eliminate_red_eyes_cpu(img_cpu.ptr<unsigned char>(), img_cpu.rows, img_cpu.cols, ul, lr, src_color, dest_color, threshold);
	stopCPU = clock();

	elapsedTimeCPU = (double)(stopCPU - startCPU) / CLOCKS_PER_SEC;
	cout << "CPU sum time = " << elapsedTimeCPU * 1000 << " ms\n";
	cout << "CPU memory throughput = " << 3 * static_cast<unsigned long long>(N) * sizeof(unsigned char) / elapsedTimeCPU / 1024 / 1024 / 1024 << " Gb/s\n";
#pragma endregion

#pragma region CUDA
	cudaMalloc(&device_img, N * sizeof(unsigned char));
	cudaMemcpy(device_img, img_gpu.ptr<unsigned char>(), N * sizeof(unsigned char), cudaMemcpyHostToDevice);

	cudaEventRecord(startCUDA, 0);
	invoke_kernel(BLK, TPB, device_img, img_gpu.rows, img_gpu.cols, ul, lr, src_color, dest_color, threshold);
	cudaEventRecord(stopCUDA, 0);
	cudaEventSynchronize(stopCUDA);

	cudaEventElapsedTime(&elapsedTimeCUDA, startCUDA, stopCUDA);
	cudaMemcpy(img_gpu.data, device_img, N * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	cudaFree(device_img);

	cout << "CUDA sum time = " << elapsedTimeCUDA << " ms\n";
	cout << "CUDA memory throughput = " << 3 * static_cast<unsigned long long>(N) * sizeof(unsigned char) / elapsedTimeCUDA / 1024 / 1024 / 1.024 << " Gb/s\n";
#pragma endregion

#pragma region Results
	imwrite(argv[2], img_cpu);
	imwrite(argv[3], img_gpu);

	int mismatched = 0;
	for (int i = 0; i < N; i++)
	{
		int a = img_cpu.data[i];
		int b = img_gpu.data[i];
		if (abs(a - b) > 3)
		{
			mismatched++;
		}
	}
	cout << "Mismatched: " << mismatched << endl;
#pragma endregion
}