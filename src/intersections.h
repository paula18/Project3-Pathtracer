// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
// Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#ifndef INTERSECTIONS_H
#define INTERSECTIONS_H

//#include <glm/glm.hpp>
#include <thrust/random.h>


#include "sceneStructs.h"
#include "cudaMat4.h"
#include "utilities.h"

// Some forward declarations
__host__ __device__ glm::vec3 getPointOnRay(ray r, float t);
__host__ __device__ glm::vec3 multiplyMV(cudaMat4 m, glm::vec4 v);
__host__ __device__ glm::vec3 getSignOfRay(ray r);
__host__ __device__ glm::vec3 getInverseDirectionOfRay(ray r);
//__host__ __device__ float boxIntersectionTest1(staticGeom box, ray r, glm::vec3& intersectionPoint, glm::vec3& normal);
//__host__ __device__ float sphereIntersectionTest(staticGeom sphere, ray r, glm::vec3& intersectionPoint, glm::vec3& normal);
//__host__ __device__ float boxIntersectionTest(glm::vec3 boxMin, glm::vec3 bMax, ray r, glm::vec3& intersectionPoint, glm::vec3& normal);
//__host__ __device__ float triangleIntersectionTest(staticGeom mesh, ray r, glm::vec3 p1, glm::vec3 p2, glm::vec3 p3, glm::vec3 faceNormal, glm::vec3& intersectionPoint, glm::vec3& intersectionNormal);
//__host__ __device__ glm::vec3 getRandomPointOnCube(staticGeom cube, float randomSeed);
	

// Handy dandy little hashing function that provides seeds for random number generation
__host__ __device__ unsigned int randomHash(unsigned int a)
{
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);
    return a;
}

// Quick and dirty epsilon check
__host__ __device__ bool epsilonCheck(float a, float b)
{
    if(fabs(fabs(a)-fabs(b)) < EPSILON)
	{
        return true;
    }
	else
	{
        return false;
    }
}

// Self explanatory
__host__ __device__ glm::vec3 getPointOnRay(ray r, float t){
  return r.origin + float(t - .0001f) * glm::normalize(r.direction);
}

// This is a workaround for GLM matrix multiplication not working properly on pre-Fermi NVIDIA GPUs.
// Multiplies a cudaMat4 matrix and a vec4 and returns a vec3 clipped from the vec4
__host__ __device__ glm::vec3 multiplyMV(cudaMat4 m, glm::vec4 v){
  glm::vec3 r(1,1,1);
  r.x = (m.x.x*v.x)+(m.x.y*v.y)+(m.x.z*v.z)+(m.x.w*v.w);
  r.y = (m.y.x*v.x)+(m.y.y*v.y)+(m.y.z*v.z)+(m.y.w*v.w);
  r.z = (m.z.x*v.x)+(m.z.y*v.y)+(m.z.z*v.z)+(m.z.w*v.w);
  return r;
}

// Gets 1/direction for a ray
__host__ __device__ glm::vec3 getInverseDirectionOfRay(ray r){
  return glm::vec3(1.0/r.direction.x, 1.0/r.direction.y, 1.0/r.direction.z);
}

// Gets sign of each component of a ray's inverse direction
__host__ __device__ glm::vec3 getSignOfRay(ray r){
  glm::vec3 inv_direction = getInverseDirectionOfRay(r);
  return glm::vec3((int)(inv_direction.x < 0), (int)(inv_direction.y < 0), (int)(inv_direction.z < 0));
}

// TODO: DONE I THINK
// Cube intersection test, return -1 if no intersection, otherwise, distance to intersection
__host__ __device__ float boxIntersectionTest(staticGeom box, ray r, glm::vec3& intersectionPoint, glm::vec3& normal)
{
	glm::vec3 ro = multiplyMV(box.inverseTransform, glm::vec4(r.origin, 1.0f));   //Transform ray into world space
	glm::vec3 rd = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f))); //Transform ray direction into world space

	ray rt; 
	rt.origin = ro; 
	rt.direction = rd; 

	glm::vec4 minPoint = glm::vec4(-0.5f, -0.5f, -0.5f, 1.0f);
	glm::vec4 maxPoint = glm::vec4(0.5f, 0.5f, 0.5f, 1.0f);

	float Tnear = -1.0f * FLT_MAX; 
	float Tfar = FLT_MAX; 
	float t1, t2; 

	//x dimension
	if (rd.x == 0)  //ray is parallel
	{
		if( abs(ro.x) < 0.5f) 
			return -1;
	}

	else
	{
		t1 = (minPoint.x - ro.x)/rd.x;
		t2 = (maxPoint.x - ro.x)/rd.x;
		
		if( max(t1, t2) < Tfar) 
			Tfar = max(t1, t2);
		if( min(t1, t2) > Tnear) 
			Tnear = min(t1, t2);
		if (Tnear > Tfar)    //Miss the box
			return -1; 
		if (Tfar < 0)
			return -1;		//Box is behind	
	}

	//y dimension
	if (rd.y == 0)  //ray is parallel
	{
		if( abs(ro.y) < 0.5f) 
			return -1;
	}

	else
	{
		t1 = (minPoint.y - ro.y)/rd.y;
		t2 = (maxPoint.y - ro.y)/rd.y;

		if( max(t1, t2) < Tfar) 
			Tfar = max(t1, t2);
		if( min(t1, t2) > Tnear) 
			Tnear = min(t1, t2);
		if (Tnear > Tfar)    //Miss the box
			return -1; 
		if (Tfar < 0)
			return -1;		//Box is behind	
	}
	
	//z dimension
	if (rd.z == 0)  //ray is parallel
	{
		if( abs(ro.z) < 0.5f) 
			return -1;
	}

	else
	{
		t1 = (minPoint.z - ro.z)/rd.z;
		t2 = (maxPoint.z - ro.z)/rd.z;
		
		if( max(t1, t2) < Tfar) 
			Tfar = max(t1, t2);
		if( min(t1, t2) > Tnear) 
			Tnear = min(t1, t2);
		if (Tnear > Tfar)    //Miss the box
			return -1; 
		if (Tfar < 0)
			return -1;		//Box is behind	
	}

	glm::vec3 intPoint = getPointOnRay(rt, Tnear); 
	float minDistance = FLT_MAX; 
	float distance; 
	glm::vec3 tempNormal;

	distance = abs(maxPoint[0] - intPoint[0]);
	if(distance <= minDistance){
		minDistance = distance; 
		tempNormal = glm::vec3(1, 0, 0); 
	}
	distance = abs(minPoint[0] - intPoint[0]);
	if(distance <= minDistance){
		minDistance = distance; 
		tempNormal = glm::vec3(-1, 0, 0); 
	}
	distance = abs(maxPoint[1] - intPoint[1]);
	if(distance <= minDistance){
		minDistance = distance; 
		tempNormal = glm::vec3(0, 1, 0); 
	}
	distance = abs(minPoint[1] - intPoint[1]);
	if(distance <= minDistance){
		minDistance = distance; 
		tempNormal = glm::vec3(0, -1, 0); 
	}
	distance = abs(maxPoint[2] - intPoint[2]);
	if(distance <= minDistance){
		minDistance = distance; 
		tempNormal = glm::vec3(0, 0, 1); 
	}
	distance = abs(minPoint[2] - intPoint[2]);
	if(distance <= minDistance){
		minDistance = distance; 
		tempNormal = glm::vec3(0, 0, -1); 
	}


	glm::vec3 realIntersectionPoint = multiplyMV(box.transform, glm::vec4(intPoint, 1.0f));    //Transform back point into object space
	intersectionPoint = realIntersectionPoint;

	normal = glm::normalize(multiplyMV(box.transform, glm::vec4(tempNormal, 0.0f)));

    return glm::length(r.origin - realIntersectionPoint);
}

// LOOK: Here's an intersection test example from a sphere. Now you just need to figure out cube and, optionally, triangle.
// Sphere intersection test, return -1 if no intersection, otherwise, distance to intersection
__host__ __device__ float sphereIntersectionTest(staticGeom sphere, ray r, glm::vec3& intersectionPoint, glm::vec3& normal)
{
  
  float radius = .5;
        
  glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin,1.0f));
  glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction,0.0f)));

  ray rt; rt.origin = ro; rt.direction = rd;
  
  float vDotDirection = glm::dot(rt.origin, rt.direction);
  float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - pow(radius, 2));
  if (radicand < 0)
  {
    return -1;
  }
  
  float squareRoot = sqrt(radicand);
  float firstTerm = -vDotDirection;
  float t1 = firstTerm + squareRoot;
  float t2 = firstTerm - squareRoot;
  
  float t = 0;
  if (t1 < 0 && t2 < 0)
  {
      return -1;
  } 
  else if (t1 > 0 && t2 > 0) 
	  t = min(t1, t2);  
  else  
	  t = max(t1, t);


  glm::vec3 realIntersectionPoint = multiplyMV(sphere.transform, glm::vec4(getPointOnRay(rt, t), 1.0));
  glm::vec3 realOrigin = multiplyMV(sphere.transform, glm::vec4(0,0,0,1));

  intersectionPoint = realIntersectionPoint;
  normal = glm::normalize(realIntersectionPoint - realOrigin);
        
  return glm::length(r.origin - realIntersectionPoint);
}


__host__ __device__ float triangleIntersectionTest(staticGeom mesh, ray r, glm::vec3 p1, glm::vec3 p2, glm::vec3 p3, glm::vec3& intersectionPoint, glm::vec3& intersectionNormal)
{

	//Calculate plane that triangle is in.
	//Equation of a plane ax+by+cz = d or n dot x = d

	//Invert the ray
	glm::vec3 ro = multiplyMV(mesh.inverseTransform, glm::vec4(r.origin,1.0f));
	glm::vec3 rd = glm::normalize(multiplyMV(mesh.inverseTransform, glm::vec4(r.direction,0.0f)));

	glm::vec3 normal = glm::normalize(glm::cross((p2-p1), (p3-p1)));

	ray rt; 
	rt.origin = ro; 
	rt.direction = rd;
	
	//Determine intersection

	float tden = glm::dot(normal, rd);

	if (tden == 0)
	{				//They are parallel
		return -1;
	}

	float d = glm::dot(normal, p1);

	float t = (-1.0f * glm::dot(normal, ro) + d)/ tden;
	
	if (t <= 0)
	{						//It is behind
		return -1;
	}

	glm::vec3 Q = ro + t*rd;  //local point of intersection

	//Check if this point is inside or outside triangle
	float firstSide = glm::dot(normal, glm::cross((p2-p1), (Q-p1)));
	float secondSide = glm::dot(normal, glm::cross((p3-p2), (Q-p2)));
	float thirdSide = glm::dot(normal, glm::cross((p1-p3), (Q-p3)));
	if(firstSide < 0 || secondSide < 0 || thirdSide < 0){
		return -1; 
	}

	glm::vec3 realIntersectionPoint = multiplyMV(mesh.transform, glm::vec4(Q, 1.0));
	glm::vec3 realOrigin = multiplyMV(mesh.transform, glm::vec4(0,0,0,1));

	intersectionPoint = realIntersectionPoint;
	intersectionNormal = -1.0f* glm::normalize(multiplyMV(mesh.transform, glm::vec4(normal,0.0f)));
        
	glm::length(r.origin - realIntersectionPoint);
}

// Returns x,y,z half-dimensions of tightest bounding box
__host__ __device__ glm::vec3 getRadiuses(staticGeom geom){
    glm::vec3 origin = multiplyMV(geom.transform, glm::vec4(0,0,0,1));
    glm::vec3 xmax = multiplyMV(geom.transform, glm::vec4(.5,0,0,1));
    glm::vec3 ymax = multiplyMV(geom.transform, glm::vec4(0,.5,0,1));
    glm::vec3 zmax = multiplyMV(geom.transform, glm::vec4(0,0,.5,1));
    float xradius = glm::distance(origin, xmax);
    float yradius = glm::distance(origin, ymax);
    float zradius = glm::distance(origin, zmax);
    return glm::vec3(xradius, yradius, zradius);
}

// LOOK: Example for generating a random point on an object using thrust.
// Generates a random point on a given cube
__host__ __device__ glm::vec3 getRandomPointOnCube(staticGeom cube, float randomSeed){

    thrust::default_random_engine rng(randomHash(randomSeed));
    thrust::uniform_real_distribution<float> u01(0,1);
    thrust::uniform_real_distribution<float> u02(-0.5,0.5);

    // Get surface areas of sides
    glm::vec3 radii = getRadiuses(cube);
    float side1 = radii.x * radii.y * 4.0f; //x-y face
    float side2 = radii.z * radii.y * 4.0f; //y-z face
    float side3 = radii.x * radii.z* 4.0f; //x-z face
    float totalarea = 2.0f * (side1+side2+side3);
    
    // Pick random face, weighted by surface area
    float russianRoulette = (float)u01(rng);
    
    glm::vec3 point = glm::vec3(.5,.5,.5);
    
    if(russianRoulette<(side1/totalarea)){
        // x-y face
        point = glm::vec3((float)u02(rng), (float)u02(rng), .5);
    }else if(russianRoulette<((side1*2)/totalarea)){
        // x-y-back face
        point = glm::vec3((float)u02(rng), (float)u02(rng), -.5);
    }else if(russianRoulette<(((side1*2)+(side2))/totalarea)){
        // y-z face
        point = glm::vec3(.5, (float)u02(rng), (float)u02(rng));
    }else if(russianRoulette<(((side1*2)+(side2*2))/totalarea)){
        // y-z-back face
        point = glm::vec3(-.5, (float)u02(rng), (float)u02(rng));
    }else if(russianRoulette<(((side1*2)+(side2*2)+(side3))/totalarea)){
        // x-z face
        point = glm::vec3((float)u02(rng), .5, (float)u02(rng));
    }else{
        // x-z-back face
        point = glm::vec3((float)u02(rng), -.5, (float)u02(rng));
    }
    
    glm::vec3 randPoint = multiplyMV(cube.transform, glm::vec4(point,1.0f));

    return randPoint;
       
}

// Generates a random point on a given sphere
__host__ __device__ glm::vec3 getRandomPointOnSphere(staticGeom sphere, float randomSeed){

	thrust::default_random_engine rng(randomHash(randomSeed));
    thrust::uniform_real_distribution<float> u01(-1,1);
    thrust::uniform_real_distribution<float> u02(0, TWO_PI);

	//Spherical coordinates for a set of points uniformly distributed over S2
	float u = (float)u01(rng);        // u = cos(phi)
	float theta = (float)u02(rng);

    float x = sqrt(1.0f - u * u) * cos(theta);      // x = sin(phi) * cos(theta)
	float y = sqrt(1.0f - u * u) * sin(theta);		// x = sin(phi) * sin(theta)
	float z = u;									// z = cos(phi)

	glm::vec3 point = glm::vec3(x, y, z); 


    glm::vec3 randPoint = multiplyMV(sphere.transform, glm::vec4(point,1.0f));

    return randPoint;
}

__host__ __device__ float geomsIntersectionTest(staticGeom* geom, int numberOfGeoms, ray& r, glm::vec3& intersectionPoint, glm::vec3& intersectionNormal, material* mat,
											int& materialIndex)
{
	float nearDistance = FLT_MAX;
	glm::vec3 tempIntersectionPoint; 
	glm::vec3 tempIntersectionNormal;
	float tempDistance = -1;
	bool hit = false;

	for (int numGeoms = 0; numGeoms < numberOfGeoms; ++numGeoms)
	{
		if (geom[numGeoms].type == CUBE)
		{
			tempDistance = boxIntersectionTest(geom[numGeoms], r, tempIntersectionPoint, tempIntersectionNormal);
		}
		else if (geom[numGeoms].type == SPHERE)
		{
			tempDistance = sphereIntersectionTest(geom[numGeoms], r, tempIntersectionPoint, tempIntersectionNormal);
		}

		else if (geom[numGeoms].type == MESH)
		{

			int numTriangles = 2*geom[numGeoms].objMesh.numberOfFaces;

			for( int i = 0; i < numTriangles; ++i)
			{
				int id1 = geom[numGeoms].objMesh.faces[i].x; 
				int id2 = geom[numGeoms].objMesh.faces[i].y; 
				int id3 = geom[numGeoms].objMesh.faces[i].z; 

				glm::vec3 p1 = geom[numGeoms].objMesh.vertex[id1];
				glm::vec3 p2 = geom[numGeoms].objMesh.vertex[id2];
				glm::vec3 p3 = geom[numGeoms].objMesh.vertex[id3];
				
				tempDistance = triangleIntersectionTest(geom[numGeoms], r, p1, p2, p3, tempIntersectionPoint, tempIntersectionNormal);

				if (tempDistance > 0 && tempDistance < nearDistance)
				{
					nearDistance = tempDistance;
					intersectionPoint = tempIntersectionPoint;
					intersectionNormal = tempIntersectionNormal;
					materialIndex = geom[numGeoms].materialid;
					hit = true;
				}
			}
		}
		if (tempDistance > 0 && tempDistance < nearDistance)
		{
			nearDistance = tempDistance;
			intersectionPoint = tempIntersectionPoint;
			intersectionNormal = tempIntersectionNormal;
			materialIndex = geom[numGeoms].materialid;
			hit = true;
		}

	}
	if(hit) 
		return nearDistance; 
	else
		return -1;
}


#endif


