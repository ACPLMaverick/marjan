// Interp.cpp : Defines the entry point for the console application.
//

#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <Windows.h>

#define MSIZE 4
#define PI 3.14159265

using namespace std;

struct Matrix
{
	float fields[MSIZE][MSIZE];
};

struct Quaternion
{
	float w, x, y, z;
};

double pcFreq = 0.0;
__int64 counterStart = 0;

void WriteMatrix(Matrix* mat)
{
	cout << fixed << showpoint;
	cout << setprecision(6);
	int cWidth = cout.width();

	cout << "-----------------------------------------------" << endl;
	for (int y = 0; y < MSIZE; ++y)
	{
		cout << "| ";
		for (int x = 0; x < MSIZE; ++x)
		{
			cout.width(10);
			cout << mat->fields[y][x];
			cout.width(cWidth);
			cout << " ";
		}
		cout << "|" << endl;
	}
	cout << "-----------------------------------------------" << endl << endl;
}

void CreateIdentity(Matrix* mat)
{
	for (int y = 0; y < MSIZE; ++y)
	{
		for (int x = 0; x < MSIZE; ++x)
		{
			if (x == y) mat->fields[y][x] = 1.0f;
			else mat->fields[y][x] = 0.0f;
		}
	}
}

void CreateZero(Matrix* mat)
{
	for (int y = 0; y < MSIZE; ++y)
	{
		for (int x = 0; x < MSIZE; ++x)
		{
			mat->fields[y][x] = 0.0f;
		}
	}
}

void CreateZero(Quaternion* q)
{
	q->w = 0.0f;
	q->x = 0.0f;
	q->y = 0.0f;
	q->z = 0.0f;
}

void CreateTranslation(Matrix* mat, float x, float y, float z)
{
	mat->fields[0][3] = x;
	mat->fields[1][3] = y;
	mat->fields[2][3] = z;
	mat->fields[3][3] = 1.0f;
}

void CreateRotationX(Matrix* mat, float rotation)
{
	mat->fields[1][1] = cos(rotation);
	mat->fields[1][2] = -sin(rotation);
	mat->fields[2][1] = sin(rotation);
	mat->fields[2][2] = cos(rotation);
}

void CreateRotationY(Matrix* mat, float rotation)
{
	mat->fields[0][0] = cos(rotation);
	mat->fields[0][2] = sin(rotation);
	mat->fields[2][0] = -sin(rotation);
	mat->fields[2][2] = cos(rotation);
}

void CreateRotationZ(Matrix* mat, float rotation)
{
	mat->fields[0][0] = cos(rotation);
	mat->fields[0][1] = -sin(rotation);
	mat->fields[1][0] = sin(rotation);
	mat->fields[1][1] = cos(rotation);
}

void MatrixMultiply(Matrix* first, Matrix* second, Matrix* out)
{
	float temp[MSIZE][MSIZE];

	for (int i = 0; i < MSIZE; ++i)
	{
		for (int j = 0; j < MSIZE; ++j)
		{
			temp[i][j] = 0.0f;
			for (int k = 0; k < MSIZE; ++k)
			{
				temp[i][j] += first->fields[i][k] * second->fields[k][j];
			}
		}
	}

	for (int i = 0; i < MSIZE; ++i)
	{
		for (int j = 0; j < MSIZE; ++j)
		{
			out->fields[i][j] = temp[i][j];
		}
	}
}

inline float Lerp(float a, float b, float amount)
{
	if (amount > 1.0f) amount = 1.0f;
	else if (amount < 0.0f) amount = 0.0f;

	return a + ((b - a) * amount);
}

inline float Pow2(float x)
{
	return x * x;
}

void GetQuaternionFromMatrix(Matrix* mat, Quaternion* q)
{
	float trace = mat->fields[0][0] + mat->fields[1][1] + mat->fields[2][2];
	float s;

	if (trace > 0.0f)
	{
		s = sqrt(trace + 1.0f) * 2.0f;
		q->w = 0.25f * s;
		q->x = (mat->fields[1][2] - mat->fields[2][1]) / s;
		q->y = (mat->fields[2][0] - mat->fields[0][2]) / s;
		q->z = (mat->fields[0][1] - mat->fields[1][0]) / s;
	}
	else if (mat->fields[0][0] > mat->fields[1][1] && mat->fields[0][0] > mat->fields[2][2])
	{
		s = sqrt(1.0f + mat->fields[0][0] - mat->fields[1][1] - mat->fields[2][2]) * 2.0f;
		q->w = (mat->fields[1][2] - mat->fields[2][1]) / s;
		q->x = 0.25f * s;;
		q->y = (mat->fields[1][0] + mat->fields[0][1]) / s;
		q->z = (mat->fields[2][0] + mat->fields[0][2]) / s;
	}
	else if (mat->fields[1][1] > mat->fields[2][2])
	{
		s = sqrt(1.0f + mat->fields[1][1] - mat->fields[0][0] - mat->fields[2][2]) * 2.0f;
		q->w = (mat->fields[2][0] - mat->fields[0][2]) / s;
		q->x = (mat->fields[1][0] + mat->fields[0][1]) / s;
		q->y = 0.25f * s;
		q->z = (mat->fields[2][1] + mat->fields[1][2]) / s;
	}
	else
	{
		s = sqrt(1.0f + mat->fields[2][2] - mat->fields[0][0] - mat->fields[1][1]) * 2.0f;
		q->w = (mat->fields[0][1] - mat->fields[1][0]) / s;
		q->x = (mat->fields[2][0] + mat->fields[0][2]) / s;
		q->y = (mat->fields[2][1] + mat->fields[1][2]) / s;
		q->z = 0.25f * s;
	}
}

void GetMatrixFromQuaternion(Quaternion* q, Matrix* mat)
{
	mat->fields[0][0] = 1.0f - 2.0f * Pow2(q->y) - 2.0f * Pow2(q->z);
	mat->fields[1][0] = 2.0f * q->x * q->y - 2.0f * q->z * q->w;
	mat->fields[2][0] = 2.0f * q->x * q->z + 2.0f * q->y * q->w;
	mat->fields[0][1] = 2.0f * q->x * q->y + 2.0f * q->z * q->w;
	mat->fields[1][1] = 1 - 2.0f * Pow2(q->x) - 2.0f * Pow2(q->z);
	mat->fields[2][1] = 2.0f * q->y * q->z - 2.0f * q->x * q->w;
	mat->fields[0][2] = 2.0f * q->x * q->z - 2.0f * q->y * q->w;
	mat->fields[1][2] = 2.0f * q->y * q->z + 2.0f * q->x * q->w;
	mat->fields[2][2] = 1 - 2.0f * Pow2(q->x) - 2.0f * Pow2(q->y);
}

void QuaternionInterpolate(Quaternion* first, Quaternion* second, const float interp, Quaternion* out)
{
	out->w = (1.0f - interp) * first->w + interp * second->w;
	out->x = (1.0f - interp) * first->x + interp * second->x;
	out->y = (1.0f - interp) * first->y + interp * second->y;
	out->z = (1.0f - interp) * first->z + interp * second->z;

	float length = sqrt(Pow2(out->w) + Pow2(out->x) + Pow2(out->y) + Pow2(out->z));

	out->w /= length;
	out->x /= length;
	out->y /= length;
	out->z /= length;
}

void MatrixInterpolate(Matrix* first, Matrix* second, const float interp, Matrix* out)
{
	// interpolating translation
	out->fields[0][3] = Lerp(first->fields[0][3], second->fields[0][3], interp);
	out->fields[1][3] = Lerp(first->fields[1][3], second->fields[1][3], interp);
	out->fields[2][3] = Lerp(first->fields[2][3], second->fields[2][3], interp);
	out->fields[3][3] = Lerp(first->fields[3][3], second->fields[3][3], interp);

	Quaternion quat1, quat2, qout;
	CreateZero(&quat1);
	CreateZero(&quat2);
	CreateZero(&qout);

	GetQuaternionFromMatrix(first, &quat1);
	GetQuaternionFromMatrix(second, &quat2);

	QuaternionInterpolate(&quat1, &quat2, interp, &qout);

	GetMatrixFromQuaternion(&qout, out);
}

float GetInterpFromConsole()
{
	float interp = -1.0f;
	string temp;
	do
	{
		cout << "Give LERP factor <0, 1>: ";
		cin >> interp;
	} while (!(interp >= 0.0f && interp <= 1.0f));
	
	return interp;
}

void StartCounter()
{
	LARGE_INTEGER li;
	QueryPerformanceFrequency(&li);

	pcFreq = double(li.QuadPart) / 1000.0;

	QueryPerformanceCounter(&li);
	counterStart = li.QuadPart;
}

double GetCounter()
{
	LARGE_INTEGER li;
	QueryPerformanceCounter(&li);
	return double(li.QuadPart - counterStart) / pcFreq;
}

int main(int argc, const char* argv[])
{
	// generating data
	Matrix mOne, mTwo, trOne, trTwo, rotXOne, rotYOne, rotXTwo, interpolated;
	double counterCreate, counterCompute;

	StartCounter();

	CreateIdentity(&mOne);
	CreateIdentity(&mTwo);

	CreateIdentity(&trOne);
	CreateIdentity(&trTwo);
	CreateIdentity(&rotXOne);
	CreateIdentity(&rotYOne);
	CreateIdentity(&rotXTwo);
	CreateZero(&interpolated);

	CreateTranslation(&trOne, 1.5f, 2.0f, 1.0f);
	CreateTranslation(&trTwo, 0.3f, 8.2f, 6.7f);
	CreateRotationX(&rotXOne, PI / 2.0f);
	CreateRotationY(&rotYOne, -PI / 4.0f);
	CreateRotationX(&rotXTwo, PI / 3.0f);

	MatrixMultiply(&rotXOne, &rotYOne, &rotXOne);
	MatrixMultiply(&trOne, &rotXOne, &trOne);
	MatrixMultiply(&mOne, &trOne, &mOne);

	MatrixMultiply(&trTwo, &rotXTwo, &trTwo);
	MatrixMultiply(&mTwo, &trTwo, &mTwo);

	counterCreate = GetCounter();

	//////////////////////////

	cout << "Before interpolation:" << endl;

	WriteMatrix(&mOne);
	WriteMatrix(&mTwo);

	float interp = GetInterpFromConsole();

	StartCounter();

	MatrixInterpolate(&mOne, &mTwo, interp, &interpolated);

	counterCompute = GetCounter();

	WriteMatrix(&interpolated);

	cout << endl << "=========================================================" << endl << endl;
	cout << "Data creating time: " << counterCreate << " ms" << endl;
	cout << "Computing time: " << counterCompute << " ms" << endl;
	cout << "Total time: " << counterCreate + counterCompute << " ms" << endl;

	getchar();
	getchar();
	return 0;
}

