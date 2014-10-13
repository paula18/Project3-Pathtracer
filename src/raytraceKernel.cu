// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Peter Kutz and Yining Karl Li's GPU Pathtracer: http://gpupathtracer.blogspot.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#include <stdio.h>
#include <cuda.h>
#include <cmath>

#include "glm/glm.hpp"
#include "utilities.h"
#include "raytraceKernel.h"
#include "intersections.h"
#include "interactions.h"

#include <thrust/device_ptr.h>
#include <thrust/partition.h>
#include <thrust/random.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/remove.h>

void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
    exit(EXIT_FAILURE); 
  }
} 

// LOOK: This function demonstrates how to use thrust for random number generation on the GPU!
// Function that generates static.
__host__ __device__ glm::vec3 generateRandomNumberFromThread(glm::vec2 resolution, float time, int x, int y){
  int index = x + (y * resolution.x);
   
  thrust::default_random_engine rng(randomHash(index*time));
  thrust::uniform_real_distribution<float> u01(0,1);

  return glm::vec3((float) u01(rng), (float) u01(rng), (float) u01(rng));
}


// Function that does the initial raycast from the camera
//__host__ __device__ ray raycastFromCameraKernel(glm::vec2 resolution, float time, int x, int y, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov){
__host__ __device__ ray raycastFromCameraKernel(glm::vec2 resolution, int time, int x, int y, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov, 
	float focal, float aperture)

{
	
	/*DOF information:
	To render each pixel:

	Start by casting a ray like normal from the eye through the pixel out into the scene. Instead of intersecting it with objects in the scene, 
	though, you just want to find the point on the ray for which the distance from the eye is equal to the selected focal distance. 
	Call this point the focal point for the pixel.
	
	Now select a random starting point on the aperture. For a circular aperture, it's pretty easy, you can select a random polar angle and a
	random radius (no greater than the radius of the aperture). You want a uniform distribution over the entire aperture, don't try to bias 
	it towards the center or anything.
	
	Cast a ray from your selected point on the aperture through the focal point. Note that it will not necessarily pass through the same pixel, that's ok.
	Render this ray the way your normally would (e.g., path tracing, or just finding the nearest point of intersection, etc.).

	Repeat steps 2, 3, and 4 some number of times, using different a random starting point on the aperture each time, 
	but always casting it through the focal point. Sum up the rendered color values from all of the rays and use that as the value for this pixel (as usual, divide by a constant attenuation factor if necessary).
	*/

	ray r;
	
	glm::vec3 A = glm::cross(view, up);
	glm::vec3 B = glm::cross(A, view);	//B is in the correct plane
	glm::vec3 M = eye + view;			    //Midpoint of screen
	glm::vec3 H = glm::normalize(A) * glm::length(view) * glm::tan(glm::radians(fov.x));
	glm::vec3 V = glm::normalize(B) * glm::length(view) * glm::tan(glm::radians(fov.y));
	
	float sx = x / ( (float)resolution.x - 1.0f);
	float sy = y / ( (float)resolution.y - 1.0f);

	glm::vec3 P = M + (2*sx - 1.0f) * H + (-2.0f * sy + 1.0f) * V;

	r.origin = eye;
	r.direction = glm::normalize(P - eye);


	//DOF calculations
	if (focal != 0)
	{
		float zDist = (focal - eye.z) / r.direction.z;
		glm::vec3 focalPoint = r.origin + r.direction*zDist;
		thrust::default_random_engine rng(randomHash(time));
		thrust::uniform_real_distribution<float> a(-1, 1); 
		thrust::uniform_real_distribution<float> b(-aperture, aperture); 

		float angle = TWO_PI * (float) a(rng); 
		float aper = (float) b(rng); 
		float dx = cos(angle) * aper; 
		float dy = sin(angle) * aper; 

		r.origin += glm::vec3(dx, dy, 0); 
		r.direction = glm::normalize(focalPoint - r.origin); 


	}
	
	return r;
}

//Kernel that blacks out a given image buffer
__global__ void clearImage(glm::vec2 resolution, glm::vec3* image){
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);

	if(x<=resolution.x && y<=resolution.y)
	{
      image[index] = glm::vec3(0,0,0);
    }
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image, int iteration){
  
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int pixelIndex = x + (y * resolution.x);
  

	if(x<=resolution.x && y<=resolution.y){

		glm::vec3 color;
		color.x = image[pixelIndex].x*255.0/ iteration;
		color.y = image[pixelIndex].y*255.0/ iteration;
		color.z = image[pixelIndex].z*255.0/ iteration;

		if(color.x>255){
	        color.x = 255;
		}

		if(color.y>255){
			color.y = 255;
		}

		if(color.z>255){
			color.z = 255;
		}
      
		// Each thread writes one pixel location in the texture (textel)
		PBOpos[pixelIndex].w = 0;
		PBOpos[pixelIndex].x = color.x;
		PBOpos[pixelIndex].y = color.y;
		PBOpos[pixelIndex].z = color.z;
	}
}


struct isDeath
{
	__host__ __device__ 
	bool operator()(const ray r) 
	{
		return !r.exist;
	}
};

struct isAlive
{
	__host__ __device__
	bool operator()(const ray r)
	{
		return r.exist;
	}
};

void motionBlur(geom* geoms, int geomID, int numOfGeoms, int iterations, int frameNumber, glm::vec3 pos, int finalT, int stepT)
{

	if(iterations < finalT)
	{
		if (iterations % stepT == 0)
		{
			geoms[geomID].translations[frameNumber] = geoms[geomID].translations[frameNumber] + pos;
			glm::mat4 newTransform = utilityCore::buildTransformationMatrix(geoms[geomID].translations[frameNumber], geoms[geomID].rotations[frameNumber], 
			geoms[geomID].scales[frameNumber]); 
			glm::mat4 newInverseTransform = glm::inverse(newTransform);
			geoms[geomID].transforms[frameNumber] = utilityCore::glmMat4ToCudaMat4(newTransform); 
			geoms[geomID].inverseTransforms[frameNumber] = utilityCore::glmMat4ToCudaMat4(newInverseTransform);
		}
	}

}

__device__ bool isLight(material m)
{
	return m.emittance > 0.0f;
}

__global__ void createRayArray(ray* rayArray, float time, cameraData cam, int hits)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y; 
	int index = x + (y * cam.resolution.x); 

	if (x > cam.resolution.x && y > cam.resolution.y)
		return;

	thrust::default_random_engine rng(randomHash(((time*index))));
	thrust::uniform_real_distribution<float> u01(-1.0, 1.0);
	float rand = (float) u01(rng);

	ray r = raycastFromCameraKernel(cam.resolution, time, x + rand, y + rand, cam.position, cam.view, cam.up, cam.fov, cam.focal, cam.aperture);
	r.exist = true; 
	r.color = glm::vec3(1.0f, 1.0f, 1.0f);
	r.pixelPos = glm::vec2(x, y);
	r.pixelIndex = index;
	rayArray[index] = r;
}

__global__ void raytraceRay(glm::vec2 resolution, float time, cameraData cam, int hits, glm::vec3* colors,
                            staticGeom* geoms, int numberOfGeoms, material* mat, ray* raysArray, int numRays) 
{

	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	if (index >= numRays)
		return;
	
	int pixIndex = raysArray[index].pixelIndex;
	
	//Rays is dead
	if (raysArray[index].exist == false)
		return;
	
	float distance;
	glm::vec3 intersectionPoint; 
	glm::vec3 intersectionNormal;
	int materialIndex; 
	float randomSeed = index * time * (hits + 1);

	//Check intersection with geometry
	distance = geomsIntersectionTest(geoms, numberOfGeoms, raysArray[index], intersectionPoint, intersectionNormal, mat, materialIndex);
		

	//If no intersection kill ray
	if (epsilonCheck(distance, -1) == true)
	{
		raysArray[index].exist = false;
		colors[pixIndex] += glm::vec3(0.0f);
		return;
	}
	
	if (isLight(mat[materialIndex]))
	{
		raysArray[index].exist = false; 
		colors[pixIndex] += raysArray[index].color * mat[materialIndex].color * mat[materialIndex].emittance ; 
		return; 
	}
	else
		calculateBSDF(resolution, raysArray[index], intersectionPoint, intersectionNormal, mat[materialIndex], randomSeed);

	
}

// TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms)
{


	int traceDepth = 4;      //determines how many bounces the raytracer traces

	// set up crucial magic
	int tileSize = 16;
    dim3 threadsPerBlock(tileSize, tileSize);
    dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));
  
  // send image to GPU
	glm::vec3* cudaimage = NULL;
	cudaMalloc((void**)&cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
	cudaMemcpy( cudaimage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);
 

  // package geometry and sent to GPU
	staticGeom* geomList = new staticGeom[numberOfGeoms];
	for(int i=0; i<numberOfGeoms; i++){
		staticGeom newStaticGeom;
		newStaticGeom.type = geoms[i].type;
		newStaticGeom.materialid = geoms[i].materialid;
		newStaticGeom.translation = geoms[i].translations[frame];
		newStaticGeom.rotation = geoms[i].rotations[frame];
		newStaticGeom.scale = geoms[i].scales[frame];
		newStaticGeom.transform = geoms[i].transforms[frame];
		newStaticGeom.inverseTransform = geoms[i].inverseTransforms[frame];
		newStaticGeom.ID = geoms[i].ID;
	
	
		if (geoms[i].type == MESH)
		{
			newStaticGeom.objMesh.numberOfFaces = geoms[i].objMesh.numberOfFaces;
			newStaticGeom.objMesh.numberOfVertices = geoms[i].objMesh.numberOfVertices;

			int numOfVert = geoms[i].objMesh.numberOfVertices; 
			cudaMalloc((void**)& (newStaticGeom.objMesh.vertex), numOfVert*sizeof(glm::vec3));
			cudaMemcpy(newStaticGeom.objMesh.vertex, &(geoms[i].objMesh.vertex[0]), numOfVert*sizeof(glm::vec3), cudaMemcpyHostToDevice);

			int numOfFaces = geoms[i].objMesh.numberOfFaces; 
			cudaMalloc((void**)&(newStaticGeom.objMesh.faces), numOfFaces*sizeof(glm::vec3));
			cudaMemcpy(newStaticGeom.objMesh.faces, &(geoms[i].objMesh.faces[0]), numOfFaces*sizeof(glm::vec3), cudaMemcpyHostToDevice);

			int numOfNormal = geoms[i].objMesh.numberOfFaces; 
			cudaMalloc((void**)&(newStaticGeom.objMesh.normal), numOfNormal*sizeof(glm::vec3));
			cudaMemcpy(newStaticGeom.objMesh.normal, &(geoms[i].objMesh.normal[0]), numOfNormal*sizeof(glm::vec3), cudaMemcpyHostToDevice);

			newStaticGeom.objMesh.numberOfFaces = geoms[i].objMesh.numberOfFaces;
			newStaticGeom.objMesh.numberOfVertices = geoms[i].objMesh.numberOfVertices;
		}
	
		geomList[i] = newStaticGeom;
	}
  
	staticGeom* cudageoms = NULL;
	cudaMalloc((void**)&cudageoms, numberOfGeoms*sizeof(staticGeom));
	cudaMemcpy( cudageoms, geomList, numberOfGeoms*sizeof(staticGeom), cudaMemcpyHostToDevice);

  // package material and sent to GPU
	material* cudamaterials = NULL;
	cudaMalloc((void**) &cudamaterials, numberOfMaterials*sizeof(material));
	cudaMemcpy( cudamaterials, materials, numberOfMaterials*sizeof(material), cudaMemcpyHostToDevice);

	// package camera
	cameraData cam;
	cam.resolution = renderCam->resolution;
	cam.position = renderCam->positions[frame];
	cam.view = renderCam->views[frame];
	cam.up = renderCam->ups[frame];
	cam.fov = renderCam->fov;
	cam.aperture = renderCam->aperture; 
	cam.focal = renderCam->focal;


	//Motion Blur
	
	//motionBlur(geoms, 5,  numberOfGeoms, iterations, frame, glm::vec3(-0.1, 0, 0), 2000, 20);
	//motionBlur(geoms, 6,  numberOfGeoms, iterations, frame, glm::vec3(0.05, 0, 0), 2000, 20);

	//Allocate memory for rays
	int numberOfRays = renderCam->resolution.x * renderCam->resolution.y;
	ray* cudarays = NULL;
	cudaMalloc((void**) &cudarays, (int)numberOfRays*sizeof(ray));

	//Create the pool of rays
	createRayArray<<<fullBlocksPerGrid, threadsPerBlock>>>(cudarays, (float)iterations, cam, traceDepth);

	// kernel launches
	for (int i = 0; i < traceDepth; ++i)
	{
		int rayThreadsPerBlock = (tileSize * tileSize);
		dim3 rayFullBlocksPerGrid((int) ceil( (float)numberOfRays/(tileSize*tileSize)));
	
		raytraceRay<<<rayFullBlocksPerGrid, rayThreadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, i, cudaimage, cudageoms, numberOfGeoms, cudamaterials, cudarays, numberOfRays);


		thrust::device_ptr<ray> devicePointer = thrust::device_pointer_cast(cudarays);
		thrust::device_ptr<ray> newDevicePointer = thrust::remove_if(devicePointer, devicePointer + numberOfRays, isDeath());
		numberOfRays = newDevicePointer.get() - devicePointer.get();

		//if there are no more rays we are done
		if (numberOfRays <= 0)
		{
			printf("Done!!");
			break;
		}
	}
 
	sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage, iterations);

	// retrieve image from GPU
	cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	// free up stuff, or else we'll leak memory like a madman
	cudaFree( cudaimage );
	cudaFree( cudageoms );
	cudaFree( cudamaterials );
	cudaFree( cudarays );

	delete[] geomList;

	// make certain the kernel has completed
	cudaThreadSynchronize();

	checkCUDAError("Kernel failed!");
}
