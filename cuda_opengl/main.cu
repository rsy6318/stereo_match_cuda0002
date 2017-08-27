#define GL_GLEXT_PROTOTYPES
#include "GL/glut.h"
#include "cuda.h"
#include "cuda_gl_interop.h"
#include "book.h"
#include "cpu_bitmap.h"

//PFNGLBINDBUFFERARBPROC    glBindBuffer     = NULL;
//PFNGLDELETEBUFFERSARBPROC glDeleteBuffers  = NULL;
//PFNGLGENBUFFERSARBPROC    glGenBuffers     = NULL;
//PFNGLBUFFERDATAARBPROC    glBufferData     = NULL;

#define DIM 512

GLuint bufferObj;   //Gpengl中的命名
cudaGraphicsResource *resource;  //CUDA中的命名

__global__ void kernel(uchar4 *ptr)
{
	unsigned int x=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int y=threadIdx.y+blockIdx.y*blockDim.y;
	unsigned int offset=x+y*blockDim.x*gridDim.x;

	float fx=x/(float)DIM-0.5;
	float fy=y/(float)DIM-0.5;
	unsigned char green=128+127*sin(abs(fx*100)-abs(fy*100));
	ptr[offset].x=0;
	ptr[offset].y=green;
	ptr[offset].z=0;
	ptr[offset].w=255;
}

static void draw_func(void)
{
	glDrawPixels(DIM,DIM,GL_RGBA,GL_UNSIGNED_BYTE,0);
	glutSwapBuffers();
}

static void key_func(unsigned char key,int x,int y)
{
	switch(key)
	{
	case 27:
		HANDLE_ERROR(cudaGraphicsUnregisterResource(resource));
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB,0);
		glDeleteBuffers(1,&bufferObj);
		exit(0);
	}
}


int main(int argc,char **argv)
{
	cudaDeviceProp prop;
	int dev;

	memset(&prop,0,sizeof(cudaDeviceProp));
	prop.major=1;
	prop.minor=0;
	HANDLE_ERROR(cudaChooseDevice(&dev,&prop));
	HANDLE_ERROR(cudaGLSetGLDevice(dev));//告诉CUDA运行的时候使用哪个设备来执行CUDA和opengl

	//执行其他操作的GL调用之前，需要首先执行这些GLUT调用
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE|GLUT_RGBA);
	glutInitWindowSize(DIM,DIM);
	glutCreateWindow("bitmap");

	//glBindBuffer    = (PFNGLBINDBUFFERARBPROC)GET_PROC_ADDRESS("glBindBuffer");
	//    glDeleteBuffers = (PFNGLDELETEBUFFERSARBPROC)GET_PROC_ADDRESS("glDeleteBuffers");
	 //   glGenBuffers    = (PFNGLGENBUFFERSARBPROC)GET_PROC_ADDRESS("glGenBuffers");
	 //   glBufferData    = (PFNGLBUFFERDATAARBPROC)GET_PROC_ADDRESS("glBufferData");

	//在Opengl中创建一个像素缓冲区对象，并将句柄保存在全局变量GLuint bufferObj中
	glGenBuffers(1,&bufferObj);  //生成一个缓冲区句柄
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB,bufferObj);   //将句柄绑定到像素缓冲区
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB,DIM*DIM*4,NULL,GL_DYNAMIC_DRAW_ARB);   //请求Opengl驱动程序来分配一个缓冲区

	//将bufferObj注册为一个图形资源
	HANDLE_ERROR(cudaGraphicsGLRegisterBuffer(	&resource,   //CUDA
												bufferObj,   //opengl
												cudaGraphicsMapFlagsNone));   //标志表示不需要为缓冲区指定特定的行为


	HANDLE_ERROR(cudaGraphicsMapResources(1,&resource,NULL));
	uchar4* devPtr;
	size_t size;
	HANDLE_ERROR(cudaGraphicsResourceGetMappedPointer((void **)&devPtr,&size,resource));

	//通过GLUT注册键盘回调函数和显示回调函数，通过glutMainLoop()将执行控制交给GLUT绘制循环。
	dim3 grids(DIM/16,DIM/16);
	dim3 threads(16,16);
	kernel<<<grids,threads>>>(devPtr);

	HANDLE_ERROR(cudaGraphicsUnmapResources(1,&resource,NULL));

	//设置好GLUT并启动循环
	glutKeyboardFunc(key_func);
	glutDisplayFunc(draw_func);
	glutMainLoop();
	return 0;
}
