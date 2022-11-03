# pragma once
# include <Eigen/Dense>
# include "global.h"

struct quaternion
{
	double s = 0.0;
	double3 v;
};

double3 operator+ (double3 a, double3 b);
double3 operator- (double3 a, double3 b);
double operator* (double3 a, double3 b);
double3 operator* (double a, double3 b);
//这是一个叉乘算符
double3 operator% (double3 a, double3 b);
double3 operator/ (double3 a, double b);


quaternion make_quaternion(double s, double i, double j, double k);
quaternion make_quaternion(double s, double3 v);

quaternion operator+ (quaternion a, quaternion b);

quaternion operator- (quaternion a, quaternion b);

quaternion operator* (double a, quaternion b);

double abs_quaternion(quaternion a);

quaternion operator* (quaternion a, quaternion b);

void norm_quaternion(quaternion& q);

void quaternion_to_matrix(quaternion q, double* R);
