# include "quaternion.h"

double3 operator+ (double3 a, double3 b)
{
	double3 result;
	result.x = a.x + b.x;
	result.y = a.y + b.y;
	result.z = a.z + b.z;
	return result;
}

double3 operator- (double3 a, double3 b)
{
	double3 result;
	result.x = a.x - b.x;
	result.y = a.y - b.y;
	result.z = a.z - b.z;
	return result;
}

double operator* (double3 a, double3 b)
{
	return (a.x * b.x + a.y * b.y + a.z * b.z);
}

double3 operator* (double a, double3 b)
{
	double3 result;
	result.x = a * b.x;
	result.y = a * b.y;
	result.z = a * b.z;
	return result;
}

//这是一个叉乘算符
double3 operator% (double3 a, double3 b)
{
	double3 result;
	result.x = a.y * b.z - a.z * b.y;
	result.y = a.z * b.x - a.x * b.z;
	result.z = a.x * b.y - a.y * b.x;
	return result;
}

double3 operator/ (double3 a, double b)
{
	double3 result;
	result.x = a.x / b;
	result.y = a.y / b;
	result.z = a.z / b;
	return result;
}

quaternion make_quaternion(double s, double i, double j, double k)
{
	quaternion x;
	x.s = s;
	//x.v = Eigen::Vector3d(i, j, k);
	x.v.x = i;
	x.v.y = j;
	x.v.z = k;
	return x;
}

quaternion make_quaternion(double s, double3 v)
{
	quaternion x;
	x.s = s;
	//x.v = Eigen::Vector3d(i, j, k);
	x.v = v;
	return x;
}

quaternion operator+ (quaternion a, quaternion b)
{
	quaternion result;
	result.s = a.s + b.s;
	result.v = a.v + b.v;
	return result;
}


quaternion operator- (quaternion a, quaternion b)
{
	quaternion result;
	result.s = a.s - b.s;
	result.v = a.v - b.v;
	return result;
}

quaternion operator* (double a, quaternion b)
{
	quaternion result;
	result.s = a * b.s;
	result.v = a * b.v;
	return result;
}

double abs_quaternion(quaternion a)
{
	//return (a.s * a.s + a.v.dot(a.v));
	return (a.s * a.s + a.v * a.v);
}

quaternion operator* (quaternion a, quaternion b)
{
	double real = a.s * b.s - a.v * b.v;
	double3 axis = a.s * b.v + b.s * a.v + a.v % b.v;
	quaternion result = make_quaternion(real, axis);
	return result;
}

void norm_quaternion(quaternion& q)
{
	double norm = abs_quaternion(q);
	q.s /= norm;
	q.v = q.v / norm;
}

void quaternion_to_matrix(quaternion Q, double* R)
{
	//Eigen::Matrix3d Rotate_Matrix;
	double s, x, y, z;
	s = Q.s;
	x = Q.v.x;
	y = Q.v.y;
	z = Q.v.z;
	R[0] = s * s + x * x - y * y - z * z;
	R[1] = 2 * (x * y - s * z);
	R[2] = 2 * (x * z + s * y);
	R[3] = 2 * (x * y + s * z);
	R[4] = s * s - x * x + y * y - z * z;
	R[5] = 2 * (y * z - s * x);
	R[6] = 2 * (x * z - s * y);
	R[7] = 2 * (y * z + s * x);
	R[8] = s * s - x * x - y * y + z * z;
}