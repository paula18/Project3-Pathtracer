/*#include "glm/glm.hpp"
//#include "cudaMat4.h"
//#include <cuda_runtime.h>
#include <string>
#include <vector>

using namespace std;


class triangle
{
public:
	glm::vec3 p1; 
	glm::vec3 p2; 
	glm::vec3 p3; 

	triangle(){}; 
	~triangle(){delete this;}
	triangle(glm::vec3 p1, glm::vec3 p2, glm::vec3 p3)
	{
		this->p1 = p1; 
		this->p2 = p2; 
		this->p3 = p3;
	}
			
	glm::vec3 getMidPoint()
	{
		glm::vec3 midPoint = (this->p1 + this->p2 + this->p3) / 3.0f; 
		return midPoint; 

	}
	

};

class boundingBox
{
public: 
	glm::vec3 boundMin; 
	glm::vec3 boundMax; 
	//int numberFaces;   //Number of faces it encloses
	
	boundingBox()
	{
		boundMin = glm::vec3(0.0f);
		boundMax = glm::vec3(1.0f);
	} 
	boundingBox(glm::vec3 vmin, glm::vec3 vmax, int numFaces);
	boundingBox(glm::vec3 p1, glm::vec3 p2, glm::vec3 p3);
	boundingBox(vector<glm::vec3> polygonVertices);
	boundingBox(vector<triangle*> &t);
	~boundingBox(){delete this;};
	
	int longestAxis();
	//float boxIntersectionTest(ray r, glm::vec3& intersectionPoint, glm::vec3& normal);
};

*/