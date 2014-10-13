// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
// Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#ifndef INTERACTIONS_H
#define INTERACTIONS_H

#include "intersections.h"
//#include "kdtree.h"


struct Fresnel {
  float reflectionCoefficient;
  float transmissionCoefficient;
};

struct AbsorptionAndScatteringProperties{
    glm::vec3 absorptionCoefficient;
    float reducedScatteringCoefficient;
};

// Forward declaration
__host__ __device__ bool calculateScatterAndAbsorption(ray& r, float& depth, AbsorptionAndScatteringProperties& currentAbsorptionAndScattering, glm::vec3& unabsorbedColor, material m, float randomFloatForScatteringDistance, float randomFloat2, float randomFloat3);
__host__ __device__ glm::vec3 getRandomDirectionInSphere(float xi1, float xi2);
__host__ __device__ glm::vec3 calculateTransmission(glm::vec3 absorptionCoefficient, float distance);
__host__ __device__ glm::vec3 calculateTransmissionDirection(glm::vec3 normal, glm::vec3 incident, float incidentIOR, float transmittedIOR);
__host__ __device__ glm::vec3 calculateReflectionDirection(glm::vec3 normal, glm::vec3 incident);
__host__ __device__ Fresnel calculateFresnel(glm::vec3 normal, glm::vec3 incident, float incidentIOR, float transmittedIOR, glm::vec3 reflectionDirection, glm::vec3 transmissionDirection);
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(glm::vec3 normal, float xi1, float xi2);

// TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
__host__ __device__ glm::vec3 calculateTransmission(glm::vec3 absorptionCoefficient, float distance) {
  return glm::vec3(0,0,0);
}

// TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
__host__ __device__ bool calculateScatterAndAbsorption(ray& r, float& depth, AbsorptionAndScatteringProperties& currentAbsorptionAndScattering,
                                                        glm::vec3& unabsorbedColor, material m, float randomFloatForScatteringDistance, float randomFloat2, float randomFloat3){
  return false;
}

// TODO (OPTIONAL): IMPLEMENT THIS FUNCTION
__host__ __device__ glm::vec3 calculateTransmissionDirection(glm::vec3 normal, glm::vec3 incident, float incidentIOR, float transmittedIOR) {
  
	
	float n12 = incidentIOR / transmittedIOR;

	glm::vec3 direction = glm::refract(incident, normal, n12);

/*	float dot = glm::dot(normal, incident);
	float a = -1.0f * n12 * dot;
	float b = 1.0f - n12 * n12 * (1 - dot * dot); 
	float root = sqrt(b);
	
	if ( b < 0.0f )
	{
		return glm::vec3(0.0f);
		//normal = -1.0f * normal; 
		//dot = -1.0f * dot; 
	}

	if (dot > 0)
	{
		normal = -1.0f * normal; 
		dot = -1.0f * dot; 
	}

		
	glm::vec3 c = (a - root) * normal; 
	glm::vec3 direction = c + n12 * incident;
	
	*/return direction;

}

// TODO: DONE
__host__ __device__ glm::vec3 calculateReflectionDirection(glm::vec3 normal, glm::vec3 incident)
{
	/*if(epsilonCheck(glm::length(glm::cross(incident, normal)), 0.0f))
		return -1.0f * normal;
	else if(epsilonCheck(glm::dot(-1.0f * incident, normal), 0.0f))
		return incident;
	else*/
	 return glm::reflect(incident, normal);//glm::normalize(incident - 2.0f*normal*glm::dot(incident, normal));
}


__host__ __device__ Fresnel calculateFresnel(glm::vec3 normal, glm::vec3 incident, float incidentIOR, float transmittedIOR)
{
	Fresnel fresnel;

	float n12 = incidentIOR / transmittedIOR; 

	float cosThetaI = -1.0f * glm::dot(incident, normal); 
	float sin2ThetaT = n12 * n12 * (1 - cosThetaI * cosThetaI); 
	float cosThetaT = sqrt(1 - sin2ThetaT);

	float a = (incidentIOR - transmittedIOR) / (incidentIOR + transmittedIOR);
	float R0 = a * a;

	float b5 = (1 - cosThetaI) * (1 - cosThetaI) * (1 - cosThetaI) * (1 - cosThetaI)
		  * (1 - cosThetaI);

	float c5 = (1 - cosThetaT) * (1 - cosThetaT) * (1 - cosThetaT) * (1 - cosThetaT) 
		  * (1 - cosThetaT); 

	float Rschlick; 
	float Tschlick; 


	//Schlick's approx
	if ( incidentIOR <= transmittedIOR)
		  Rschlick = R0 + (1 - R0) * b5; 
	
	else if ( (incidentIOR > transmittedIOR) && (sin2ThetaT <= 1.0f) )
		  Rschlick = R0 + (1 - R0) * c5; 
	  
	else if ( (incidentIOR > transmittedIOR) && (sin2ThetaT > 1.0f) )
		  Rschlick = 1; 

	Tschlick = 1 - Rschlick; 

	
	fresnel.reflectionCoefficient = Rschlick;
	fresnel.transmissionCoefficient = Tschlick;
	
	return fresnel;
}

// LOOK: This function demonstrates cosine weighted random direction generation in a sphere!
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(glm::vec3 normal, float xi1, float xi2) {
    
    // Crucial difference between this and calculateRandomDirectionInSphere: THIS IS COSINE WEIGHTED!
    
    float up = sqrt(xi1); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = xi2 * TWO_PI;
    
    // Find a direction that is not the normal based off of whether or not the normal's components are all equal to sqrt(1/3) or whether or not at least one component is less than sqrt(1/3). Learned this trick from Peter Kutz.
    
    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
      directionNotNormal = glm::vec3(1, 0, 0);
    } else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
      directionNotNormal = glm::vec3(0, 1, 0);
    } else {
      directionNotNormal = glm::vec3(0, 0, 1);
    }
    
    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 = glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 = glm::normalize(glm::cross(normal, perpendicularDirection1));
    
    return ( up * normal ) + ( cos(around) * over * perpendicularDirection1 ) + ( sin(around) * over * perpendicularDirection2 );
    
}


// Now that you know how cosine weighted direction generation works, try implementing 
// non-cosine (uniform) weighted random direction generation.
// This should be much easier than if you had to implement calculateRandomDirectionInHemisphere.
__host__ __device__ glm::vec3 getRandomDirectionInSphere(float xi1, float xi2) {

	float alpha = xi2 * TWO_PI; 
	
	float z = 1.0f - 2.0f  * xi1; 
	float x = cos(alpha) * sqrt(1 - z * z);
	float y = sin(alpha) * sqrt(1 - z * z);

	return glm::vec3(x, y, z);
}


__host__ __device__ int calculateBSDF(glm::vec2 resolution, ray& r, glm::vec3 intersectionPoint, glm::vec3 intersectionNormal, material m, float randomSeed)
{
	thrust::default_random_engine rng(randomHash(randomSeed));
	thrust::uniform_real_distribution<float> u01(0,1);
	thrust::uniform_real_distribution<float> u02(0,1);
	thrust::uniform_real_distribution<float> u03(0,1);
	thrust::uniform_real_distribution<float> u04(0,1);
	float random = (float)u01(rng); 
	float random2 = (float)u02(rng);
	float random3 = (float)u03(rng);
	float random4 = (float)u04(rng);
	
	
	// Diffuse case
	if(!m.hasReflective && !m.hasRefractive)
	{
		glm::vec3 outputDir = calculateRandomDirectionInHemisphere(intersectionNormal, random, random2);
			
		r.direction = glm::normalize(outputDir);
		r.origin = intersectionPoint  + 1e-3f * r.direction;
		r.color *= m.color;
		return 0;
	}

	// Reflective case
	else if(m.hasReflective && !m.hasRefractive)
	{
		
		float diffuseExponent = 0.8f;

		if (random3 < diffuseExponent)
		{
			r.direction = glm::normalize(calculateRandomDirectionInHemisphere(intersectionNormal, random, random2));
			r.origin = intersectionPoint  + 1e-3f * r.direction; 

			r.color *= m.color;

			return 0;
				
		}
		else
		{
			r.direction = calculateReflectionDirection(intersectionNormal, r.direction);
			r.origin = intersectionPoint  + 1e-3f * r.direction; 
			r.color *= m.specularColor; 		
		
			return 1;
		
		}
	}

	//Refractive clase
	else if(m.hasRefractive)
	{
	
		Fresnel fresnelCoefficient; 
		bool inside; 

		//Ray is outside 
		if(glm::dot(r.direction, intersectionNormal) < 0)
		{
			inside = false; 
			fresnelCoefficient = calculateFresnel(r.direction, intersectionNormal, 1.0f, m.indexOfRefraction); 
		}
		//Ray is inside
		else
		{
			inside = true;
			fresnelCoefficient = calculateFresnel(r.direction, -1.0f* intersectionNormal, m.indexOfRefraction, 1.0f);
		}

		if (random4 < fresnelCoefficient.reflectionCoefficient)
		{

			r.direction = calculateReflectionDirection(intersectionNormal, r.direction);
			r.origin = intersectionPoint  + 1e-3f * r.direction; 

			r.color *= m.specularColor; 		
			return 1;
		
		}
		else 
		{
			if (inside)
				r.direction = calculateTransmissionDirection(-intersectionNormal, r.direction, m.indexOfRefraction, 1.0f);
			else
				r.direction = calculateTransmissionDirection(intersectionNormal, r.direction, 1.0f, m.indexOfRefraction);
			
			r.origin = intersectionPoint  + 1e-3f * r.direction; 
			return 2 ;		
		}	
	}
}

#endif
