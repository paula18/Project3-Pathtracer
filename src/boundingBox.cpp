/*#include "boundingBox.h"


boundingBox::boundingBox(glm::vec3 vmin, glm::vec3 vmax, int numFaces)
{
	boundMin = vmin; 
	boundMax = vmax; 
}

boundingBox::boundingBox(vector<glm::vec3> polygonVertices)
{
	glm::vec3 minPoint = glm::vec3(1000000.0f); 
	glm::vec3 maxPoint = glm::vec3(-10000000.0f); 

	for (int i = 0; i < polygonVertices.size(); ++i)
	{
		this->boundMin = glm::vec3(min(minPoint.x, polygonVertices[i].x), min(minPoint.y, polygonVertices[i].y), min(minPoint.z, polygonVertices[i].z));
		this->boundMax = glm::vec3(max(maxPoint.x, polygonVertices[i].x), max(maxPoint.y, polygonVertices[i].y), max(maxPoint.z, polygonVertices[i].z));		
	}

}

boundingBox::boundingBox(glm::vec3 p1, glm::vec3 p2, glm::vec3 p3)
{
	glm::vec3 minPoint = glm::vec3(0.0f); 
	glm::vec3 maxPoint = glm::vec3(0.0f); 

	if ( p1.x < minPoint.x) minPoint.x = p1.x; 
	if ( p1.y < minPoint.y) minPoint.y = p1.y;
	if ( p1.z < minPoint.z) minPoint.z = p1.z;

	if ( p1.x > maxPoint.x) maxPoint.x = p1.x; 
	if ( p1.y > maxPoint.y) maxPoint.y = p1.y;
	if ( p1.z > maxPoint.z) maxPoint.z = p1.z;

	if ( p2.x < minPoint.x) minPoint.x = p2.x; 
	if ( p2.y < minPoint.y) minPoint.y = p2.y;
	if ( p2.z < minPoint.z) minPoint.z = p2.z;

	if ( p2.x > maxPoint.x) maxPoint.x = p2.x; 
	if ( p2.y > maxPoint.y) maxPoint.y = p2.y;
	if ( p2.z > maxPoint.z) maxPoint.z = p2.z;

	if ( p3.x < minPoint.x) minPoint.x = p3.x; 
	if ( p3.y < minPoint.y) minPoint.y = p3.y;
	if ( p3.z < minPoint.z) minPoint.z = p3.z;

	if ( p3.x > maxPoint.x) maxPoint.x = p3.x; 
	if ( p3.y > maxPoint.y) maxPoint.y = p3.y;
	if ( p3.z > maxPoint.z) maxPoint.z = p3.z;

	this->boundMax = maxPoint; 
	this->boundMax = minPoint; 
}

boundingBox::boundingBox(std::vector<triangle*> &t)
{

	glm::vec3 minPoint = glm::vec3(1000000.0f); 
	glm::vec3 maxPoint = glm::vec3(-10000000.0f); 
	
	for (int i = 0; i < t.size(); ++i)
	{
		boundingBox*  b = new boundingBox(t[i]->p1, t[i]->p2, t[i]->p3);
		this->boundMin = glm::vec3(min(minPoint.x, b->boundMin.x), min(minPoint.y, b->boundMin.y), min(minPoint.z, b->boundMin.z));
		this->boundMax = glm::vec3(max(maxPoint.x, b->boundMax.x), max(maxPoint.y, b->boundMax.y), max(maxPoint.z, b->boundMax.z));

	}

}


int boundingBox::longestAxis()
{
	float longAxis = -1000000.0f;
	float axisX = this->boundMax.x - this->boundMin.x; 
	float axisY = this->boundMax.y - this->boundMin.y;
	float axisZ = this->boundMax.z - this->boundMin.z;

	if(axisX > longAxis)
	{
		longAxis = axisX;
		return 0;
	}
	else if(axisY > longAxis)
	{
		longAxis = axisY;
		return 1; 
	}
	else if(axisZ > longAxis)
	{
		longAxis = axisZ; 
		return 2;
	}

	
}
/*
float boundingBox::boxIntersectionTest(ray r, glm::vec3& intersectionPoint, glm::vec3& normal)
{
	//glm::vec3 ro = multiplyMV(box.inverseTransform, glm::vec4(r.origin, 1.0f));   //Transform ray into world space
	//glm::vec3 rd = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f))); //Transform ray direction into world space

//	ray rt; 
//	rt.origin = ro; 
//	rt.direction = rd; 

	glm::vec4 minPoint = glm::vec4(this->boundMax, 1.0f);
	glm::vec4 maxPoint = glm::vec4(this->boundMax, 1.0f);

	float Tnear = -10000000000.0f; 
	float Tfar = 10000000000.0f; 
	float t1, t2; 

	int dimension = 0; 
	 
	while (dimension < 3)
	{
		if (r.direction[dimension] == 0)  //ray is parallel
		{
			if( (r.origin[dimension] < minPoint[dimension]) || (r.origin[dimension] > maxPoint[dimension]) )
				return -1;
		}

		else
		{
			t1 = (minPoint[dimension] - r.origin[dimension])/r.direction[dimension];
			t2 = (maxPoint[dimension] - r.origin[dimension])/r.direction[dimension];

			if (t1 > t2)
			{
				float temp = t1;
				t1 = t2; 
				t2 = temp; 
			}
			if (t1 > Tnear)
				Tnear = t1;
			if (t2 < Tfar)
				Tfar = t2;
			if (Tnear > Tfar)    //Miss the box
				return -1; 
			if (Tfar < 0)
				return -1;		//Box is behind	
		}

		dimension++;
	}

	glm::vec3 intPoint = r.origin + float(Tnear - .0001f) * glm::normalize(r.direction);
	float minDistance = 10000; 
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

//	glm::vec3 realIntersectionPoint = multiplyMV(box.transform, glm::vec4(intPoint, 1.0f));    //Transform back point into object space
//	intersectionPoint = realIntersectionPoint;

	intersectionPoint = intPoint; 
	normal = glm::normalize(tempNormal);//glm::normalize(multiplyMV(box.transform, glm::vec4(tempNormal, 0.0f)));

    return glm::length(r.origin - intPoint);
}

*/