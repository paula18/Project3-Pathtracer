/*#ifndef KDTREE_H
#define KDTREE_H

//#include "glm/glm.hpp"
//#include "cudaMat4.h"
//#include <cuda_runtime.h>
#include <string>
#include <vector>

#include "boundingBox.h"

///Info from: http://blog.frogslayer.com/kd-trees-for-faster-ray-tracing-with-triangles/
class Node
{
public:
	boundingBox bbox; 
	Node* left; 
	Node* right; 
	std::vector<triangle*> triangles; 

	Node()
	{
		bbox = boundingBox(); 
		left = NULL; 
		right = NULL;
	}
	~Node(){delete this;}

	Node* build(std::vector<triangle*>& t, int depth); 
	//bool hit(staticGeom mesh, Node* node, ray &r, glm::vec3 &interPoint, glm::vec3 &interNormal, float &d);



};

#endif; 
*/