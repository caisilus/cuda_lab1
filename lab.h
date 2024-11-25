#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>

struct RGB {
	unsigned char r;
	unsigned char g;
	unsigned char b;

	__host__ __device__ float dist(unsigned char r, unsigned char g, unsigned char b) const
	{
		int r1 = r;
		int r2 = this->r;

		int g1 = g;
		int g2 = this->g;

		int b1 = b;
		int b2 = this->b;

		return sqrtf((r1 - r2) * (r1 - r2) + (g1 - g2) * (g1 - g2) + (b1 - b2) * (b1 - b2));
	}
};

struct Coord
{
	int x;
	int y;
};

void eliminate_red_eyes_cpu(unsigned char* img, size_t rows, size_t cols, Coord ul, Coord lr, RGB src_color, RGB dest_color, float threshold);

void invoke_kernel(int BLK, int TPB, unsigned char* img, size_t rows, size_t cols, Coord ul, Coord lr, RGB src_color, RGB dest_color, float threshold);