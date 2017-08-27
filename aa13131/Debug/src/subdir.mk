################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/boxFilter.cpp \
../src/boxFilter_cpu.cpp 

CU_SRCS += \
../src/boxFilter_kernel.cu 

CU_DEPS += \
./src/boxFilter_kernel.d 

OBJS += \
./src/boxFilter.o \
./src/boxFilter_cpu.o \
./src/boxFilter_kernel.o 

CPP_DEPS += \
./src/boxFilter.d \
./src/boxFilter_cpu.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -I"/usr/local/cuda-8.0/samples/3_Imaging" -I"/usr/local/cuda-8.0/samples/common/inc" -I"/home/rsy6318/cuda-workspace/aa13131" -G -g -O0 -gencode arch=compute_30,code=sm_30  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -I"/usr/local/cuda-8.0/samples/3_Imaging" -I"/usr/local/cuda-8.0/samples/common/inc" -I"/home/rsy6318/cuda-workspace/aa13131" -G -g -O0 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -I"/usr/local/cuda-8.0/samples/3_Imaging" -I"/usr/local/cuda-8.0/samples/common/inc" -I"/home/rsy6318/cuda-workspace/aa13131" -G -g -O0 -gencode arch=compute_30,code=sm_30  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -I"/usr/local/cuda-8.0/samples/3_Imaging" -I"/usr/local/cuda-8.0/samples/common/inc" -I"/home/rsy6318/cuda-workspace/aa13131" -G -g -O0 --compile --relocatable-device-code=false -gencode arch=compute_30,code=compute_30 -gencode arch=compute_30,code=sm_30  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


