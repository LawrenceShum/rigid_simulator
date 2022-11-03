# pragma once
# include <cuda_runtime.h>
# include "device_launch_parameters.h"
# include "helper_cuda.h" 
# include "helper_functions.h"
# include "cublas_v2.h"
# include "cusparse.h"
# include "cuda.h"
# include "cuda_device_runtime_api.h"
# include "cuda_runtime_api.h"



# define nThreadMax_perBlock 512

# define FLIP