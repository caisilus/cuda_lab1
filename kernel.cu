#include "lab.h"

__global__ void eliminate_red_eyes_gpu(unsigned char* img, size_t rows, size_t cols, Coord ul, Coord lr, RGB src_color, RGB dest_color, float threshold)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int threadsNum = blockDim.x * gridDim.x;

	for (int i = (id + ul.y * cols + ul.x) * 3; i < (lr.y * cols + lr.x) * 3; i += threadsNum * 3)
	{
		/*img[i] = dest_color.r;
		img[i + 1] = dest_color.g;
		img[i + 2] = dest_color.b;*/
		if (src_color.dist(img[i], img[i + 1], img[i + 2]) < threshold)
		{
			img[i] = dest_color.r;
			img[i + 1] = dest_color.g;
			img[i + 2] = dest_color.b;
		}
	}
}

void invoke_kernel(int BLK, int TPB, unsigned char* img, size_t rows, size_t cols, Coord ul, Coord lr, RGB src_color, RGB dest_color, float threshold)
{
	eliminate_red_eyes_gpu <<<BLK, TPB>>> (img, rows, cols, ul, lr, src_color, dest_color, threshold);
}

