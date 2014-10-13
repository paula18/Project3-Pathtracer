/*#include "kdtree.h"

#include <iostream>

Node* Node::build(std::vector<triangle*>& t, int depth)
{

	std::cout << "Building trees" << std::endl;
	Node* node = new Node(); 
/*	node->triangles = t; 
	node->left = NULL; 
	node->right = NULL; 
	node->bbox = boundingBox(); 

	if (t.size() == 0)
		return node;

	if (t.size() == 1)
	{
		node->bbox = boundingBox(t);//->get_bounding_box(); 
		node->left = new Node(); 
		node->right = new Node(); 
		node->left->triangles = std::vector<triangle*>(); 
		node->right->triangles = std::vector<triangle*>(); 
		return node; 
	}

	//Get bounding box surrounding whole object
	node->bbox = boundingBox(t); 

	/*for (int i = 1; i < t.size(); ++i)
	{
		node->bbox.expand(t[i]->get_bounding_box());
	}*/

/*	glm::vec3 midPoint = glm::vec3(0, 0, 0); 
	for (int i = 0; i < t.size(); ++i)
		//find middle point for all triangles
		midPoint = midPoint + (t[i]->getMidPoint() * (1.0f / t.size()));

	std::vector <triangle*> left_triangles; 
	std::vector <triangle*> right_triangles; 
	int axis = node->bbox.longestAxis();  

	for (int i = 0; i < t.size(); ++i)
	{
		switch(axis)
		{
		case 0:
			midPoint.x >= t[i]->getMidPoint().x ? right_triangles.push_back(t[i]) :
				left_triangles.push_back(t[i]);
			break;
		case 1:
			midPoint.y >= t[i]->getMidPoint().y ? right_triangles.push_back(t[i]) :
				left_triangles.push_back(t[i]);
			break;
		case 2:
			midPoint.z >= t[i]->getMidPoint().z ? right_triangles.push_back(t[i]) :
				left_triangles.push_back(t[i]);
			break;
		}
	}

	if(left_triangles.size() == 0 && right_triangles.size() > 0)
		left_triangles = right_triangles; 
	if(right_triangles.size() == 0 && left_triangles.size() > 0)
		right_triangles = left_triangles; 

	//If 50% match stop dividng
	int matches = 0; 
	for (int i = 0; i < left_triangles.size(); ++i)
	{
		for (int j = 0; j < right_triangles.size(); ++ j) 
		{
			if (left_triangles[i] == right_triangles[j])
				matches++; 
		}
	}

	if ((float) matches / left_triangles.size() < 0.5 && (float)matches / right_triangles.size() < 0.5f)
	{
		node->left = build(left_triangles, depth + 1); 
		node->right = build(right_triangles, depth + 1);
	}
	else
	{
		node->left = new Node(); 
		node->right = new Node(); 
		node->left->triangles = std::vector<triangle*>(); 
		node->right->triangles = std::vector<triangle*>(); 
	} 
	
	return node; 
}

*/

