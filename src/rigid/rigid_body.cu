# pragma once
# include "rigid_body.cuh"
# include "rigid_geometry.cuh"
# include <sstream>
# include <iostream>

quaternion quaternion_add(quaternion a, quaternion b)
{
	quaternion result = make_quaternion(0.0, 0.0, 0.0, 0.0);
	result.s = a.s + b.s;
	result.v.x = a.v.x + b.v.x;
	result.v.y = a.v.y + b.v.y;
	result.v.z = a.v.z + b.v.z;
	return result;
}

quaternion quaternion_multiply(quaternion a, quaternion b)
{
	quaternion result = make_quaternion(0.0, 0.0, 0.0, 0.0);
	result.s = a.s * b.s - (a.v.x * b.v.x + a.v.y * b.v.y + a.v.z * b.v.z);
	result.v.x = a.s * b.v.x + b.s * a.v.x + a.v.y * b.v.z - a.v.z * b.v.y;
	result.v.y = a.s * b.v.y + b.s * a.v.y + a.v.z * b.v.x - a.v.x * b.v.z;
	result.v.z = a.s * b.v.z + b.s * a.v.z + a.v.x * b.v.y - a.v.y * b.v.x;
	return result;
}

void Rigid_Body::determineLayout(dim3& gridLayout, dim3& blockLayout, int num)
{
	if (num <= nThreadMax_perBlock)
	{
		gridLayout = 1;
		blockLayout = num;
	}
	else
	{
		int split = num / nThreadMax_perBlock;
		gridLayout = split + 1;
		blockLayout = nThreadMax_perBlock;
	}
}

void Rigid_Geometry::load_mesh(const std::string file_name)
{
	std::ifstream in;
	in.open(file_name, std::ifstream::in);
	if (in.fail()) return;
	std::string line;
	while (!in.eof())
	{
		std::getline(in, line);
		std::istringstream iss(line.c_str());
		char trash;
		if (!line.compare(0, 2, "v "))
		{
			iss >> trash;
			double x, y, z;
			iss >> x >> y >> z;
			vertices.push_back(x);
			vertices.push_back(y);
			vertices.push_back(z);
		}
		else if (!line.compare(0, 2, "f "))
		{
			int f, t, n;
			iss >> trash;
			iss >> f >> t >> n;
			face.push_back(f - 1);
			face.push_back(t - 1);
			face.push_back(n - 1);
		}
	}
	in.close();
	num_of_vertices = vertices.size() / 3;
	vertices_in_cpu = new double[num_of_vertices * 3];
	for (int i = 0; i < vertices.size(); i++)
	{
		vertices_in_cpu[i] = vertices[i];
	}
	copy_to_device();
	std::cout << "total vertices = " << num_of_vertices << ", faces = " << face.size() / 3 << std::endl;
}

void Rigid_Geometry::copy_to_device()
{
	checkCudaErrors(cudaMalloc((void**)&vertices_in_gpu, sizeof(double) * num_of_vertices * 3));
	checkCudaErrors(cudaMalloc((void**)&vertices_in_gpu_this_step, sizeof(double) * num_of_vertices * 3));
	checkCudaErrors(cudaMemcpy(vertices_in_gpu, vertices_in_cpu, sizeof(double) * num_of_vertices * 3, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(vertices_in_gpu_this_step, vertices_in_cpu, sizeof(double) * num_of_vertices * 3, cudaMemcpyHostToDevice));
}

int Rigid_Geometry::get_num_of_vertices()
{
	return num_of_vertices;
}

double* Rigid_Geometry::get_vertices_device()
{
	return vertices_in_gpu;
}

double* Rigid_Geometry::get_vertices_cpu()
{
	return vertices_in_cpu;
}

__global__ void update_vertices_gpu(double3 x, double* rotate_matrix, double* vertices, double* output, int num)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < num)
	{
		double3 vertex = make_type3(vertices[3 * index + 0], vertices[3 * index + 1], vertices[3 * index + 2]);
		//先旋转
		double3 after_rotate;
		after_rotate.x = rotate_matrix[0] * vertex.x + rotate_matrix[1] * vertex.y + rotate_matrix[2] * vertex.z;
		after_rotate.y = rotate_matrix[3] * vertex.x + rotate_matrix[4] * vertex.y + rotate_matrix[5] * vertex.z;
		after_rotate.z = rotate_matrix[6] * vertex.x + rotate_matrix[7] * vertex.y + rotate_matrix[8] * vertex.z;
		double3 after_translation;
		after_translation.x = after_rotate.x + x.x;
		after_translation.y = after_rotate.y + x.y;
		after_translation.z = after_rotate.z + x.z;
		output[3 * index + 0] = after_translation.x;
		output[3 * index + 1] = after_translation.y;
		output[3 * index + 2] = after_translation.z;
	}
}

void Rigid_Geometry::update_vertices_location(double3 X, quaternion Q)
{
	dim3 gridLayout, blockLayout;
	if (num_of_vertices <= nThreadMax_perBlock)
	{
		gridLayout = 1;
		blockLayout = num_of_vertices;
	}
	else
	{
		gridLayout = (num_of_vertices / nThreadMax_perBlock) + 1;
		blockLayout = nThreadMax_perBlock;
	}
	double* R = new double[9];
	quaternion_to_matrix(Q, R);
	double* matrix_temp;
	checkCudaErrors(cudaMalloc((void**)&matrix_temp, sizeof(double) * 9));
	checkCudaErrors(cudaMemcpy(matrix_temp, R, sizeof(double) * 9, cudaMemcpyHostToDevice));
	update_vertices_gpu << <gridLayout, blockLayout >> >
		(X, matrix_temp, vertices_in_gpu, vertices_in_gpu_this_step, num_of_vertices);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

void Rigid_Geometry::copy_to_opengl_buffer(double* opengl_buffer)
{
	checkCudaErrors(cudaMemcpy(opengl_buffer, vertices_in_gpu_this_step, sizeof(double) * num_of_vertices * 3, cudaMemcpyDeviceToHost));
}

void Rigid_Body::set_timestep(double timestep)
{
	this->dt = timestep;
}

/*****************************translational_move*****************************/
void Rigid_Body::compute_translational_force()
{
	int num_vertices = geometry->get_num_of_vertices();
	translational_force.x = 0.0;
	translational_force.y = 0.0;
	translational_force.z = 0.0;
	for (int i = 0; i < num_vertices; i++)
	{
		double3 temp = forces_cpu[i];
		translational_force.x += temp.x;
		translational_force.y += temp.y;
		translational_force.z += temp.z;
	}
	double linear_decay = 0.999;
	double gravity = -9.8;
	velocity.x *= linear_decay;
	velocity.y *= linear_decay;
	velocity.z *= linear_decay;
	velocity.x += dt * (1.0 / mass) * translational_force.x;
	velocity.y += dt * ((1.0 / mass) * translational_force.y + gravity);
	velocity.z += dt * (1.0 / mass) * translational_force.z;

}

void Rigid_Body::translational_move()
{
	X.x += dt * velocity.x;
	X.y += dt * velocity.y;
	X.z += dt * velocity.z;
}
/**************************************************************************/

/*******************************计算力矩*******************************/
__global__ void compute_torque_elements_gpu(
	double* torque_elements, 
	double3* force_element,
	double* reference_vertices,
	double* rotate_matrix,
	int num)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	double3 vertex = make_type3(reference_vertices[index * 3 + 0], reference_vertices[index * 3 + 1], reference_vertices[index * 3 + 2]);
	if (index < num)
	{
		double3 torque;
		double3 force = force_element[index];
		double3 left;
		left.x = rotate_matrix[0] * vertex.x + rotate_matrix[1] * vertex.y + rotate_matrix[2] * vertex.z;
		left.y = rotate_matrix[3] * vertex.x + rotate_matrix[4] * vertex.y + rotate_matrix[5] * vertex.z;
		left.z = rotate_matrix[6] * vertex.x + rotate_matrix[7] * vertex.y + rotate_matrix[8] * vertex.z;
		torque.x = left.y * force.z - left.z * force.y;
		torque.y = left.x * force.z - left.z * force.x;
		torque.z = left.x * force.y - left.y * force.x;
		torque_elements[3 * index + 0] = torque.x;
		torque_elements[3 * index + 1] = torque.y;
		torque_elements[3 * index + 2] = torque.z;
	}
}

void Rigid_Body::compute_torque()
{
	int num_vertices = geometry->get_num_of_vertices();
	double* reference_vertices = geometry->get_vertices_device();
	double* temp_R;
	checkCudaErrors(cudaMalloc((void**)&temp_R, sizeof(double) * 9));
	checkCudaErrors(cudaMemcpy(temp_R, Rotate_Matrix, sizeof(double) * 9, cudaMemcpyHostToDevice));

	//thrust::device_vector<double3> torque_element(num_vertices);
	double* torque_element;
	checkCudaErrors(cudaMalloc((void**)&torque_element, sizeof(double) * num_vertices * 3));

	dim3 gridLayout, blockLayout;
	determineLayout(gridLayout, blockLayout, num_vertices);
	compute_torque_elements_gpu << <gridLayout, blockLayout >> >
		(torque_element, forces_gpu, reference_vertices, temp_R, num_vertices);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	double* tt = new double[num_vertices * 3];
	checkCudaErrors(cudaMemcpy(tt, torque_element, sizeof(double) * 3 * num_vertices, cudaMemcpyDeviceToHost));
	double3 sum_of_torque = make_type3(0.0, 0.0, 0.0);
	for (int i = 0; i < num_vertices; i++)
	{
		sum_of_torque.x += tt[i * 3 + 0];
		sum_of_torque.y += tt[i * 3 + 1];
		sum_of_torque.z += tt[i * 3 + 2];
	}
	T.x = sum_of_torque.x;
	T.y = sum_of_torque.y;
	T.z = sum_of_torque.z;
	checkCudaErrors(cudaFree(temp_R));
	checkCudaErrors(cudaFree(torque_element));

	double angular_decay = 0.999;
	Eigen::Matrix3d RR;
	RR(0, 0) = Rotate_Matrix[0];	RR(0, 1) = Rotate_Matrix[1];	RR(0, 2) = Rotate_Matrix[2];
	RR(1, 0) = Rotate_Matrix[3];	RR(1, 1) = Rotate_Matrix[4];	RR(1, 2) = Rotate_Matrix[5];
	RR(2, 0) = Rotate_Matrix[6];	RR(2, 1) = Rotate_Matrix[7];	RR(2, 2) = Rotate_Matrix[8];
	Eigen::Matrix3d II;
	II(0, 0) = inertia[0];	II(0, 1) = inertia[1];	II(0, 2) = inertia[2];
	II(1, 0) = inertia[3];	II(1, 1) = inertia[4];	II(1, 2) = inertia[5];
	II(2, 0) = inertia[6];	II(2, 1) = inertia[7];	II(2, 2) = inertia[8];
	Eigen::Matrix3d right = II * RR.transpose();
	Eigen::Matrix3d I = RR * right;
	Eigen::Vector3d TT(T.x, T.y, T.z);
	Eigen::Vector3d mid = dt * I.inverse() * TT;
	angular_velocity.x *= angular_decay;
	angular_velocity.y *= angular_decay;
	angular_velocity.z *= angular_decay;
	angular_velocity.x += mid[0];
	angular_velocity.y += mid[1];
	angular_velocity.z += mid[2];

}
/***********************************************************************/
__global__ void compute_force_gpu(double3* force, int num)
{
	int index = threadIdx.x + blockIdx.x + blockDim.x;
	double gravity_acceleration = -9.8;
	if (index >= num)
		return;
	force[index].y = 0.01 * gravity_acceleration;
	force[index].x = 0.0;
	force[index].z = 0.0;
}

void Rigid_Body::compute_force()
{
	int num = geometry->get_num_of_vertices();
	dim3 gridLayout, blockLayout;
	//determineLayout(gridLayout, blockLayout, num);
	//compute_force_gpu << <gridLayout, blockLayout >> >
	//	(forces_gpu, num);
	//checkCudaErrors(cudaGetLastError());
	//checkCudaErrors(cudaDeviceSynchronize());
	//checkCudaErrors(cudaMemcpy(forces_cpu, forces_gpu, sizeof(double3) * num, cudaMemcpyDeviceToHost));
	for (int i = 0; i < num; i++)
	{
		forces_cpu[i].x = 0.0;
		forces_cpu[i].y = 0.0;
		forces_cpu[i].z = 0.0;
	}
	//checkCudaErrors(cudaMemcpy(forces_gpu, forces_cpu, sizeof(double3) * num, cudaMemcpyHostToDevice));
}

void Rigid_Body::rotational_move()
{
	Q = Q + make_quaternion(0.0, 0.5 * dt * angular_velocity) * Q;
	norm_quaternion(Q);
}

void Rigid_Body::compute_inertia()
{
	mass = 0.0;
	int num = geometry->get_num_of_vertices();
	double* vertices = geometry->get_vertices_cpu();
	double vertex_weight = 0.01;
	Eigen::Matrix3d II = Eigen::Matrix3d::Zero();
	for (int i = 0; i < num; i++)
	{
		Eigen::Vector3d r_i(vertices[3 * i + 0], vertices[3 * i + 1], vertices[3 * i + 2]);
		II += vertex_weight * (r_i.dot(r_i) * Eigen::Matrix3d::Identity() - r_i * r_i.transpose());
		mass += vertex_weight;
	}
	for (int row = 0; row < 3; row++)
	{
		for (int col = 0; col < 3; col++)
		{
			inertia[row * 3 + col] = II(row, col);
		}
	}
	std::cout << II << std::endl;
}

void Rigid_Body::step_forward(double dt)
{
	set_timestep(dt);
	quaternion_to_matrix(Q, Rotate_Matrix);
	compute_force();
	compute_translational_force();
	compute_torque();
	detect_collision_ground();
	detect_collision_left();
	detect_collision_right();
	detect_collision_back();
	detect_collision_front();
	translational_move();
	rotational_move();
	geometry->update_vertices_location(X, Q);
}

double collision_down(double x, double y, double z, bool& flag)
{
	double phi = y - -0.5;
	if (phi < 0.0)
		flag = TRUE;
	return phi;
}

double collision_up(double x, double y, double z, bool& flag)
{
	double phi = y - 1.0;
	if (phi < 0.0)
		flag = TRUE;
	return phi;
}

double collision_left(double x, double y, double z, bool& flag)
{
	double phi = x - -1.0;
	if (phi < 0.0)
		flag = TRUE;
	return phi;
}

double collision_right(double x, double y, double z, bool& flag)
{
	double phi = x - 1.0;
	if (phi < 0.0)
		flag = TRUE;
	return phi;
}

double collision_front(double x, double y, double z, bool& flag)
{
	double phi = z - 1.0;
	if (phi < 0.0)
		flag = TRUE;
	return phi;
}

double collision_back(double x, double y, double z, bool& flag)
{
	double phi = z - -1.0;
	if (phi < 0.0)
		flag = TRUE;
	return phi;
}

void Rigid_Body::detect_collision_ground()
{
	//暂时先测试在地面的碰撞
	//地面为y=-1
	//一些碰撞参数
	double mu_N = 0.95;
	double mu_T = 1 - mu_N;
	int num = geometry->get_num_of_vertices();
	double* vertices = new double[num * 3];
	double* reference_location = geometry->get_vertices_cpu();
	geometry->copy_to_opengl_buffer(vertices);

	int count = 1;
	Eigen::Vector3d average_impulse = Eigen::Vector3d::Zero();
	Eigen::Vector3d average_r_i = Eigen::Vector3d::Zero();
	Eigen::Vector3d N(0.0, 1.0, 0.0);
	Eigen::Matrix3d R;
	R << Rotate_Matrix[0], Rotate_Matrix[1], Rotate_Matrix[2],
		Rotate_Matrix[3], Rotate_Matrix[4], Rotate_Matrix[5],
		Rotate_Matrix[6], Rotate_Matrix[7], Rotate_Matrix[8];
	Eigen::Matrix3d II;
	II << inertia[0], inertia[1], inertia[2],
		inertia[3], inertia[4], inertia[5],
		inertia[6], inertia[7], inertia[8];
	for (int index = 0; index < num; index++)
	{
		double x, y, z;
		x = vertices[3 * index + 0];
		y = vertices[3 * index + 1];
		z = vertices[3 * index + 2];
		bool is_collide = FALSE;
		//检测有没有碰到
		double phi_down = collision_down(x, y, z, is_collide);
		if (!is_collide)
			continue;
		
		Eigen::Vector3d r_i(reference_location[3 * index + 0], reference_location[3 * index + 1], reference_location[3 * index + 2]);
		Eigen::Vector3d v(velocity.x, velocity.y, velocity.z);
		Eigen::Vector3d w(angular_velocity.x, angular_velocity.y, angular_velocity.z);
		Eigen::Vector3d v_i = (v + w.cross(R * r_i));
		if (v_i.dot(N) >= 0.0)
			continue;
		Eigen::Vector3d v_N_i = (v_i.dot(N)) * N;
		Eigen::Vector3d v_T_i = v_i - v_N_i;
		double alpha = std::max(1 - mu_T * (1 + mu_N) * (v_N_i.norm() / v_T_i.norm()), 0.0);
		Eigen::Vector3d v_i_new = ( - mu_N * v_N_i + alpha * v_T_i);
		//计算冲量
		Eigen::Vector3d mul = R * r_i;
		Eigen::Matrix3d cross_product_matrix;
		cross_product_matrix << 
			0.0, -mul[2], mul[1],
			mul[2], 0.0, -mul[0],
			-mul[1], mul[0], 0.0;
		Eigen::Matrix3d K = (1.0 / mass) * Eigen::Matrix3d::Identity() - cross_product_matrix * II.inverse() * cross_product_matrix;
		average_impulse += K.inverse() * (v_i_new - v_i);
		average_r_i += r_i;
		count++;
	}
	average_impulse /= count;
	average_r_i /= count;
	velocity.x += (1.0 / mass) * average_impulse[0];
	velocity.y += (1.0 / mass) * average_impulse[1];
	velocity.z += (1.0 / mass) * average_impulse[2];
	Eigen::Vector3d w_delta = II.inverse() * ((R * average_r_i).cross(average_impulse));
	angular_velocity.x += w_delta[0];
	angular_velocity.y += w_delta[1];
	angular_velocity.z += w_delta[2];
}

void Rigid_Body::detect_collision_left()
{
	//暂时先测试在地面的碰撞
	//地面为y=-1
	//一些碰撞参数
	double mu_N = 0.9;
	double mu_T = 1 - mu_N;
	int num = geometry->get_num_of_vertices();
	double* vertices = new double[num * 3];
	double* reference_location = geometry->get_vertices_cpu();
	geometry->copy_to_opengl_buffer(vertices);

	int count = 1;
	Eigen::Vector3d average_impulse = Eigen::Vector3d::Zero();
	Eigen::Vector3d average_r_i = Eigen::Vector3d::Zero();
	Eigen::Vector3d N(1.0, 0.0, 0.0);
	Eigen::Matrix3d R;
	R << Rotate_Matrix[0], Rotate_Matrix[1], Rotate_Matrix[2],
		Rotate_Matrix[3], Rotate_Matrix[4], Rotate_Matrix[5],
		Rotate_Matrix[6], Rotate_Matrix[7], Rotate_Matrix[8];
	Eigen::Matrix3d II;
	II << inertia[0], inertia[1], inertia[2],
		inertia[3], inertia[4], inertia[5],
		inertia[6], inertia[7], inertia[8];
	for (int index = 0; index < num; index++)
	{
		double x, y, z;
		x = vertices[3 * index + 0];
		y = vertices[3 * index + 1];
		z = vertices[3 * index + 2];
		bool is_collide = FALSE;
		//检测有没有碰到
		double phi_down = collision_left(x, y, z, is_collide);
		if (!is_collide)
			continue;

		Eigen::Vector3d r_i(reference_location[3 * index + 0], reference_location[3 * index + 1], reference_location[3 * index + 2]);
		Eigen::Vector3d v(velocity.x, velocity.y, velocity.z);
		Eigen::Vector3d w(angular_velocity.x, angular_velocity.y, angular_velocity.z);
		Eigen::Vector3d v_i = (v + w.cross(R * r_i));
		if (v_i.dot(N) >= 0.0)
			continue;
		Eigen::Vector3d v_N_i = (v_i.dot(N)) * N;
		Eigen::Vector3d v_T_i = v_i - v_N_i;
		double alpha = std::max(1 - mu_T * (1 + mu_N) * (v_N_i.norm() / v_T_i.norm()), 0.0);
		Eigen::Vector3d v_i_new = (-mu_N * v_N_i + alpha * v_T_i);
		//计算冲量
		Eigen::Vector3d mul = R * r_i;
		Eigen::Matrix3d cross_product_matrix;
		cross_product_matrix <<
			0.0, -mul[2], mul[1],
			mul[2], 0.0, -mul[0],
			-mul[1], mul[0], 0.0;
		Eigen::Matrix3d K = (1.0 / mass) * Eigen::Matrix3d::Identity() - cross_product_matrix * II.inverse() * cross_product_matrix;
		average_impulse += K.inverse() * (v_i_new - v_i);
		average_r_i += r_i;
		count++;
	}
	average_impulse /= count;
	average_r_i /= count;
	velocity.x += (1.0 / mass) * average_impulse[0];
	velocity.y += (1.0 / mass) * average_impulse[1];
	velocity.z += (1.0 / mass) * average_impulse[2];
	Eigen::Vector3d w_delta = II.inverse() * ((R * average_r_i).cross(average_impulse));
	angular_velocity.x += w_delta[0];
	angular_velocity.y += w_delta[1];
	angular_velocity.z += w_delta[2];
}

void Rigid_Body::detect_collision_right()
{
	//暂时先测试在地面的碰撞
	//地面为y=-1
	//一些碰撞参数
	double mu_N = 0.9;
	double mu_T = 1 - mu_N;
	int num = geometry->get_num_of_vertices();
	double* vertices = new double[num * 3];
	double* reference_location = geometry->get_vertices_cpu();
	geometry->copy_to_opengl_buffer(vertices);

	int count = 1;
	Eigen::Vector3d average_impulse = Eigen::Vector3d::Zero();
	Eigen::Vector3d average_r_i = Eigen::Vector3d::Zero();
	Eigen::Vector3d N(-1.0, 0.0, 0.0);
	Eigen::Matrix3d R;
	R << Rotate_Matrix[0], Rotate_Matrix[1], Rotate_Matrix[2],
		Rotate_Matrix[3], Rotate_Matrix[4], Rotate_Matrix[5],
		Rotate_Matrix[6], Rotate_Matrix[7], Rotate_Matrix[8];
	Eigen::Matrix3d II;
	II << inertia[0], inertia[1], inertia[2],
		inertia[3], inertia[4], inertia[5],
		inertia[6], inertia[7], inertia[8];
	for (int index = 0; index < num; index++)
	{
		double x, y, z;
		x = vertices[3 * index + 0];
		y = vertices[3 * index + 1];
		z = vertices[3 * index + 2];
		bool is_collide = FALSE;
		//检测有没有碰到
		double phi_down = collision_right(x, y, z, is_collide);
		if (!is_collide)
			continue;

		Eigen::Vector3d r_i(reference_location[3 * index + 0], reference_location[3 * index + 1], reference_location[3 * index + 2]);
		Eigen::Vector3d v(velocity.x, velocity.y, velocity.z);
		Eigen::Vector3d w(angular_velocity.x, angular_velocity.y, angular_velocity.z);
		Eigen::Vector3d v_i = (v + w.cross(R * r_i));
		if (v_i.dot(N) >= 0.0)
			continue;
		Eigen::Vector3d v_N_i = (v_i.dot(N)) * N;
		Eigen::Vector3d v_T_i = v_i - v_N_i;
		double alpha = std::max(1 - mu_T * (1 + mu_N) * (v_N_i.norm() / v_T_i.norm()), 0.0);
		Eigen::Vector3d v_i_new = (-mu_N * v_N_i + alpha * v_T_i);
		//计算冲量
		Eigen::Vector3d mul = R * r_i;
		Eigen::Matrix3d cross_product_matrix;
		cross_product_matrix <<
			0.0, -mul[2], mul[1],
			mul[2], 0.0, -mul[0],
			-mul[1], mul[0], 0.0;
		Eigen::Matrix3d K = (1.0 / mass) * Eigen::Matrix3d::Identity() - cross_product_matrix * II.inverse() * cross_product_matrix;
		average_impulse += K.inverse() * (v_i_new - v_i);
		average_r_i += r_i;
		count++;
	}
	average_impulse /= count;
	average_r_i /= count;
	velocity.x += (1.0 / mass) * average_impulse[0];
	velocity.y += (1.0 / mass) * average_impulse[1];
	velocity.z += (1.0 / mass) * average_impulse[2];
	Eigen::Vector3d w_delta = II.inverse() * ((R * average_r_i).cross(average_impulse));
	angular_velocity.x += w_delta[0];
	angular_velocity.y += w_delta[1];
	angular_velocity.z += w_delta[2];
}

void Rigid_Body::detect_collision_back()
{
	//暂时先测试在地面的碰撞
	//地面为y=-1
	//一些碰撞参数
	double mu_N = 0.9;
	double mu_T = 1 - mu_N;
	int num = geometry->get_num_of_vertices();
	double* vertices = new double[num * 3];
	double* reference_location = geometry->get_vertices_cpu();
	geometry->copy_to_opengl_buffer(vertices);

	int count = 1;
	Eigen::Vector3d average_impulse = Eigen::Vector3d::Zero();
	Eigen::Vector3d average_r_i = Eigen::Vector3d::Zero();
	Eigen::Vector3d N(0.0, 0.0, 1.0);
	Eigen::Matrix3d R;
	R << Rotate_Matrix[0], Rotate_Matrix[1], Rotate_Matrix[2],
		Rotate_Matrix[3], Rotate_Matrix[4], Rotate_Matrix[5],
		Rotate_Matrix[6], Rotate_Matrix[7], Rotate_Matrix[8];
	Eigen::Matrix3d II;
	II << inertia[0], inertia[1], inertia[2],
		inertia[3], inertia[4], inertia[5],
		inertia[6], inertia[7], inertia[8];
	for (int index = 0; index < num; index++)
	{
		double x, y, z;
		x = vertices[3 * index + 0];
		y = vertices[3 * index + 1];
		z = vertices[3 * index + 2];
		bool is_collide = FALSE;
		//检测有没有碰到
		double phi_down = collision_back(x, y, z, is_collide);
		if (!is_collide)
			continue;

		Eigen::Vector3d r_i(reference_location[3 * index + 0], reference_location[3 * index + 1], reference_location[3 * index + 2]);
		Eigen::Vector3d v(velocity.x, velocity.y, velocity.z);
		Eigen::Vector3d w(angular_velocity.x, angular_velocity.y, angular_velocity.z);
		Eigen::Vector3d v_i = (v + w.cross(R * r_i));
		if (v_i.dot(N) >= 0.0)
			continue;
		Eigen::Vector3d v_N_i = (v_i.dot(N)) * N;
		Eigen::Vector3d v_T_i = v_i - v_N_i;
		double alpha = std::max(1 - mu_T * (1 + mu_N) * (v_N_i.norm() / v_T_i.norm()), 0.0);
		Eigen::Vector3d v_i_new = (-mu_N * v_N_i + alpha * v_T_i);
		//计算冲量
		Eigen::Vector3d mul = R * r_i;
		Eigen::Matrix3d cross_product_matrix;
		cross_product_matrix <<
			0.0, -mul[2], mul[1],
			mul[2], 0.0, -mul[0],
			-mul[1], mul[0], 0.0;
		Eigen::Matrix3d K = (1.0 / mass) * Eigen::Matrix3d::Identity() - cross_product_matrix * II.inverse() * cross_product_matrix;
		average_impulse += K.inverse() * (v_i_new - v_i);
		average_r_i += r_i;
		count++;
	}
	average_impulse /= count;
	average_r_i /= count;
	velocity.x += (1.0 / mass) * average_impulse[0];
	velocity.y += (1.0 / mass) * average_impulse[1];
	velocity.z += (1.0 / mass) * average_impulse[2];
	Eigen::Vector3d w_delta = II.inverse() * ((R * average_r_i).cross(average_impulse));
	angular_velocity.x += w_delta[0];
	angular_velocity.y += w_delta[1];
	angular_velocity.z += w_delta[2];
}

void Rigid_Body::detect_collision_front()
{
	//暂时先测试在地面的碰撞
	//地面为y=-1
	//一些碰撞参数
	double mu_N = 0.9;
	double mu_T = 1 - mu_N;
	int num = geometry->get_num_of_vertices();
	double* vertices = new double[num * 3];
	double* reference_location = geometry->get_vertices_cpu();
	geometry->copy_to_opengl_buffer(vertices);

	int count = 1;
	Eigen::Vector3d average_impulse = Eigen::Vector3d::Zero();
	Eigen::Vector3d average_r_i = Eigen::Vector3d::Zero();
	Eigen::Vector3d N(0.0, 0.0, -1.0);
	Eigen::Matrix3d R;
	R << Rotate_Matrix[0], Rotate_Matrix[1], Rotate_Matrix[2],
		Rotate_Matrix[3], Rotate_Matrix[4], Rotate_Matrix[5],
		Rotate_Matrix[6], Rotate_Matrix[7], Rotate_Matrix[8];
	Eigen::Matrix3d II;
	II << inertia[0], inertia[1], inertia[2],
		inertia[3], inertia[4], inertia[5],
		inertia[6], inertia[7], inertia[8];
	for (int index = 0; index < num; index++)
	{
		double x, y, z;
		x = vertices[3 * index + 0];
		y = vertices[3 * index + 1];
		z = vertices[3 * index + 2];
		bool is_collide = FALSE;
		//检测有没有碰到
		double phi_down = collision_front(x, y, z, is_collide);
		if (!is_collide)
			continue;

		Eigen::Vector3d r_i(reference_location[3 * index + 0], reference_location[3 * index + 1], reference_location[3 * index + 2]);
		Eigen::Vector3d v(velocity.x, velocity.y, velocity.z);
		Eigen::Vector3d w(angular_velocity.x, angular_velocity.y, angular_velocity.z);
		Eigen::Vector3d v_i = (v + w.cross(R * r_i));
		if (v_i.dot(N) >= 0.0)
			continue;
		Eigen::Vector3d v_N_i = (v_i.dot(N)) * N;
		Eigen::Vector3d v_T_i = v_i - v_N_i;
		double alpha = std::max(1 - mu_T * (1 + mu_N) * (v_N_i.norm() / v_T_i.norm()), 0.0);
		Eigen::Vector3d v_i_new = (-mu_N * v_N_i + alpha * v_T_i);
		//计算冲量
		Eigen::Vector3d mul = R * r_i;
		Eigen::Matrix3d cross_product_matrix;
		cross_product_matrix <<
			0.0, -mul[2], mul[1],
			mul[2], 0.0, -mul[0],
			-mul[1], mul[0], 0.0;
		Eigen::Matrix3d K = (1.0 / mass) * Eigen::Matrix3d::Identity() - cross_product_matrix * II.inverse() * cross_product_matrix;
		average_impulse += K.inverse() * (v_i_new - v_i);
		average_r_i += r_i;
		count++;
	}
	average_impulse /= count;
	average_r_i /= count;
	velocity.x += (1.0 / mass) * average_impulse[0];
	velocity.y += (1.0 / mass) * average_impulse[1];
	velocity.z += (1.0 / mass) * average_impulse[2];
	Eigen::Vector3d w_delta = II.inverse() * ((R * average_r_i).cross(average_impulse));
	angular_velocity.x += w_delta[0];
	angular_velocity.y += w_delta[1];
	angular_velocity.z += w_delta[2];
}