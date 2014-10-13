CUDA Pathtracer
===================

## OVERVIEW

This project is a Monte Carlo simulated pathtracer implemented in CUDA to achieve paralellism. The method works as follows: 

We trace a ray from the camera to each pixel. If the ray hits an object, we calculate the intersection point and the intersection normal, as well as the direction 
of the outgoing ray. Depending on the surface of this object we compute the accumulated color. If the ray does not hit anything we kill it. If it hits a light, we calculate the color 
that comes from it and kill it as well. We set a maximum number of times the ray can bounce. 
The main features of my path tracer are the following: 

* Rendering of diffuse surfaces.
* Rendering of specular reflective surfaces and refractive surfaces.
* Cube, sphere and triangle intersections.
* Stream compaction for optimization. 
* Depth of field.
* OBJ mesh loading and rendering. 
* Translational motion blur. 
* Jittered supersample to avoid aliasing. 
* Interactive camera. 

## FEATURES EXPLANATION

* Diffuse surfaces work as follows: the direction from an outgoing ray from a diffuse surface is calculated randomly, since diffuse surfaces have equal probability of 
reflecting light in any direction.

* Reflection and refraction were calculated using Fresnel's equation and Schlick's approximation. The amount of light reflected or refracted depend on the index of refraction 
of the material.

* Stream compaction was implemented using thrust. This algorithm speeds up the run time in great amount. 

* Depth of field: The DOF calculation is based on a focal length and aperture of the camera. Instead of
raycasting from the camera into the scene, we find a a point on the ray for which the distance from the eye is 
equal to the focal distance. Then, we select as the ray origin a random point on the aperture. 
	
* OBJ Mesh loading and rendering: Since I have not implemented bounding boxes or KD tree, currently my path tracer only allows simple OBJ meshes. Still, cool images can 
be generated. 

* Translational motion blur: motion blur is performed by translating the object a certain amount each iteration.  

* Super sampling anti-aliasing: anti-aliasing was implemented my jittering the camera randomly at each iteration. 

* The interactive camera works as follows: 
	keys up/down move the camera in the +/- y direction.
	keys left/right move the camera in the -/+ x direction.
	keys right shift/right control moves the camera in the +/- z direction.
	keys D/F increase/descrease the focal distance by 1.
	key S saves th current rendered image. 


## SCREENSHOTS

The objects in this scene have the three basic surfaces: diffuse, reflective and refractive. 


Here I included depth of field. The camera is focused on the middle scene. There was a bug with my box-ray intersection function that I did not realize before rendering this
image. That is why the top of the back box is a little darker. But that bug has been fixed!

This image includes motion blur. The green sphere is a mix of diffuse and reflective surfaces. I set a threshold and generate a random number. If this number is below the threshold, 
diffuse coloring is added to the accumulated color, otherwise, reflection is taken into account. 

Here you can see the OBJ loading and rendering feature. 

Here is a cool image of a bug. It happened when I changed the random seed when generating the direction of the outgoing ray from a diffuse surface. 

## PERFORMANCE ANALISIS

To analyse the perfomance of my path tracer I looked at two variables: the tile size and the number of bounces. 
As the table shows and as expected, the number of bounces increases the runtime. The tile size does help for a better performance. However the best option 
is to use a 16 tile size. Higher lead to a slow down of the code. 



