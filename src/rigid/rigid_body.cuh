# pragma once
# include "rigid_geometry.cuh"
# include "quaternion.h"
# include <Eigen/Dense>
# include <thrust/host_vector.h>
# include <thrust/device_vector.h>
# include <memory>


class Rigid_Body
{
private:
	double density;
	double mass;
	const std::string mesh_file_path;
	double dt;
	//thrust::device_vector<double3> reference_vertices_gpu;
	//thrust::device_vector<double3> reference_vertices_cpu;

	std::shared_ptr<Rigid_Geometry> geometry;
	double3 velocity;
	double3 angular_velocity;
	//double angularMomentum;
	double3* forces_gpu;
	double3* forces_cpu;
	double3 translational_force;

	double* inertia;
	double* inertia_inv;
	quaternion Q;
	double3 T;
	double* Rotate_Matrix;
	double3 X;
	void determineLayout(dim3& gridLayout, dim3& blockLayout, int num);

	void compute_inertia();
	void compute_force();
	void compute_translational_force();
	void compute_torque();
	void translational_move();
	void rotational_move();

	//¹ØÓÚÅö×²(penalty force)


public:
	Rigid_Body(double density, const std::string mesh_file_path)
		:density(density), mesh_file_path(mesh_file_path)
	{
		geometry = std::make_shared<Rigid_Geometry>(mesh_file_path);
		int num_vertices = geometry->get_num_of_vertices();
		checkCudaErrors(cudaMalloc((void**)&forces_gpu, sizeof(double3) * num_vertices));
		forces_cpu = new double3[num_vertices];
		for (int i = 0; i < num_vertices; i++)
		{
			forces_cpu[i].x = 0.0;
			forces_cpu[i].y = 0.0;
			forces_cpu[i].z = 0.0;
		}
		checkCudaErrors(cudaMemcpy(forces_gpu, forces_cpu, sizeof(double3) * num_vertices, cudaMemcpyHostToDevice));
		velocity = make_type3(0.0, 0.0, 0.0);
		X = make_type3(0.0, 0.0, 0.0);
		T = make_type3(0.0, 0.0, 0.0);
		angular_velocity = make_type3(0.0, 0.0, 0.0);
		Q = make_quaternion(1.0, 0.0, 0.0, 0.0);
		inertia = new double[9];
		inertia_inv = new double[9];
		Rotate_Matrix = new double[9];
		compute_inertia();
	};

	~Rigid_Body()
	{
		checkCudaErrors(cudaFree(forces_gpu));
		delete forces_cpu;
	}
	
	void step_forward(double dt);
	void set_timestep(double timestep);
	
	void copy_to_opengl_buffer(double* ptr)
	{
		geometry->copy_to_opengl_buffer(ptr);
	}
	int access_num_vertices()
	{
		return geometry->get_num_of_vertices();
	}
	int access_num_faces()
	{
		return geometry->get_num_of_faces();
	}
	void access_indices(int* opengl_buffer)
	{
		geometry->get_indices(opengl_buffer);
	}
}; 