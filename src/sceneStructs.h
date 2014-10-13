// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#ifndef CUDASTRUCTS_H
#define CUDASTRUCTS_H

#include "glm/glm.hpp"
#include "cudaMat4.h"
#include <cuda_runtime.h>
#include <string>
#include <vector>

enum GEOMTYPE{ SPHERE, CUBE, MESH };

using namespace std;


struct ray 
{
	glm::vec3 origin;
	glm::vec3 direction;
	glm::vec3 color;
	bool exist;
	glm::vec2 pixelPos;
	int pixelIndex;
};

struct obj
{

	std::vector<glm::vec3> vertex; 
	std::vector<glm::vec3> normal;
	std::vector<glm::vec3> faces; 

	int numberOfVertices; 
	int numberOfFaces;

};

struct cudaobj
{

	glm::vec3* vertex; 
	glm::vec3* normal;
	glm::vec3* faces; 

	int numberOfVertices; 
	int numberOfFaces;

};

struct geom 
{
	enum GEOMTYPE type;
	obj objMesh; 
	int materialid;
	int frames;
	glm::vec3* translations;
	glm::vec3* rotations;
	glm::vec3* scales;
	cudaMat4* transforms;
	cudaMat4* inverseTransforms;
	int ID; 
};

struct staticGeom 
{
	enum GEOMTYPE type;
	cudaobj objMesh; 
	int materialid;
	glm::vec3 translation;
	glm::vec3 rotation;
	glm::vec3 scale;
	cudaMat4 transform;
	cudaMat4 inverseTransform;
	int ID; 
};

struct cameraData 
{
	glm::vec2 resolution;
	glm::vec3 position;
	glm::vec3 view;
	glm::vec3 up;
	glm::vec2 fov;
	float aperture;
	float focal;
};

struct camera 
{
	glm::vec2 resolution;
	glm::vec3* positions;
	glm::vec3* views;
	glm::vec3* ups;
	int frames;
	glm::vec2 fov;
	unsigned int iterations;
	glm::vec3* image;
	ray* rayList;
	std::string imageName;
	float aperture;
	float focal;
};

struct material
{
	enum MATERIALTYPE mattype;
	glm::vec3 color;
	float specularExponent;
	glm::vec3 specularColor;
	float hasReflective;
	float hasRefractive;
	float indexOfRefraction;
	float hasScatter;
	glm::vec3 absorptionCoefficient;
	float reducedScatterCoefficient;
	float emittance;
};



#endif //CUDASTRUCTS_H
