################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../main.cu 

CU_DEPS += \
./main.d 

OBJS += \
./main.o 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -I/home/rsy6318/opencv-3.0.0/include/opencv -I"/usr/local/cuda-8.0/samples/common/inc" -I"/home/rsy6318/cuda-workspace/stereo_match0004" -I/home/rsy6318/opencv-3.0.0/include/opencv2 -O3 -gencode arch=compute_30,code=sm_30  -odir "." -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -I/home/rsy6318/opencv-3.0.0/include/opencv -I"/usr/local/cuda-8.0/samples/common/inc" -I"/home/rsy6318/cuda-workspace/stereo_match0004" -I/home/rsy6318/opencv-3.0.0/include/opencv2 -O3 --compile --relocatable-device-code=false -gencode arch=compute_30,code=compute_30 -gencode arch=compute_30,code=sm_30  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


