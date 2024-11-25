#include "lab.h"

void eliminate_red_eyes_cpu(unsigned char* img, size_t rows, size_t cols, Coord ul, Coord lr, RGB src_color, RGB dest_color, float threshold)
{
	for (size_t i = (ul.y * cols + ul.x) * 3; i < (lr.y * cols + lr.x) * 3; i += 3)
	{
		if (src_color.dist(img[i], img[i + 1], img[i + 2]) < threshold)
		{
			img[i] = dest_color.r;
			img[i + 1] = dest_color.g;
			img[i + 2] = dest_color.b;
		}
	}
}

