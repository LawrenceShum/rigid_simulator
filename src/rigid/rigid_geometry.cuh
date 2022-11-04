# pragma once
# include "cuda_header.cuh"
# include <vector>
# include "global.h"
# include <thrust/host_vector.h>
# include <thrust/device_vector.h>
# include "quaternion.h"

struct m_vector3
{
	double elements[3];
};

struct m_vector2
{
	double elements[2];
};

class Rigid_Geometry
{
public:
	Rigid_Geometry(const std::string mesh_file_name)
	{
		load_mesh(mesh_file_name);
	}

	~Rigid_Geometry() 
	{
		checkCudaErrors(cudaFree(vertices_in_gpu));
		checkCudaErrors(cudaFree(vertices_in_gpu_this_step));
	}

	bool is_inside();

	double get_signed_distance();

	void project_out(double3);

	void get_sample_points(double3* points);

	double get_mass(double density);

	void get_inertial_tensor(double density, double& matrix);

	void get_segments();

	void get_vertices(double3* verts);

	void load_mesh(const std::string);

	void update_vertices_location(double3, quaternion);

	//初始点
	//reference的
	thrust::host_vector<double> vertices;
	double* vertices_in_cpu;
	//reference的
	double* vertices_in_gpu;
	//更新的
	double* vertices_in_gpu_this_step;

	std::vector<int> face;
	int num_of_vertices;

	void copy_to_device();
	int get_num_of_vertices();
	int get_num_of_faces()
	{
		return face.size() / 3;
	}
	double* get_vertices_device();
	double* get_vertices_cpu();
	void copy_to_opengl_buffer(double* opengl_buffer);
	void get_indices(int* opengl_buffer)
	{
		int num = get_num_of_faces();
		for (int i = 0; i < num * 3; i++)
		{
			opengl_buffer[i] = face[i];
		}
	}
};