// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Varun Sampath and Patrick Cozzi for GLSL Loading, from CIS565 Spring 2012 HW5 at the University of Pennsylvania: http://cis565-spring-2012.github.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#include "main.h"

#include <windows.h>
#include <gl/GL.h>
#include <gl/GLU.h>
#include <gl/glut.h>


#define GLEW_STATIC


//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv){
  #ifdef __APPLE__
    // Needed in OSX to force use of OpenGL3.2 
    glfwWindowHint(GLFW_OPENGL_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_OPENGL_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  #endif 

  // Set up pathtracer stuff
  bool loadedScene = false;
  finishedRender = false;

  targetFrame = 0;
  singleFrameMode = false;

  // Load scene file
  for(int i=1; i<argc; i++){
    string header; string data;
    istringstream liness(argv[i]);
    getline(liness, header, '='); getline(liness, data, '=');
    if(strcmp(header.c_str(), "scene")==0){
      renderScene = new scene(data);
      loadedScene = true;
    }else if(strcmp(header.c_str(), "frame")==0){
      targetFrame = atoi(data.c_str());
      singleFrameMode = true;
    }
  }

  if(!loadedScene){
    cout << "Error: scene file needed!" << endl;
    return 0;
  }

  // Set up camera stuff from loaded pathtracer settings
  iterations = 0;
  renderCam = &renderScene->renderCam;
  width = renderCam->resolution[0];
  height = renderCam->resolution[1];

  if(targetFrame >= renderCam->frames){
    cout << "Warning: Specified target frame is out of range, defaulting to frame 0." << endl;
    targetFrame = 0;
  }

  // Initialize CUDA and GL components
  if (init(argc, argv)) {
    // GLFW main loop

	mainLoop();
  }

  return 0;
}

void mainLoop() {
  while(!glfwWindowShouldClose(window)){
    glfwPollEvents();
    runCuda();

    string title = "CIS565 Render | " + utilityCore::convertIntToString(iterations) + " Iterations ";
	glfwSetWindowTitle(window, title.c_str());

	    
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBindTexture(GL_TEXTURE_2D, displayImage);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glClear(GL_COLOR_BUFFER_BIT);   

    // VAO, shader program, and texture already bound
    glDrawElements(GL_TRIANGLES, 6,  GL_UNSIGNED_SHORT, 0);
    glfwSwapBuffers(window);

	}
  glfwDestroyWindow(window);
  glfwTerminate();
}

//-------------------------------
//---------RUNTIME STUFF---------
//-------------------------------

void runCuda(){

  // Map OpenGL buffer object for writing from CUDA on a single GPU
  // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer
  
  if(iterations < renderCam->iterations){

	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float time; 
	cudaEventRecord(start, 0);

    uchar4 *dptr=NULL;
    iterations++;
    cudaGLMapBufferObject((void**)&dptr, pbo);
  
    // pack geom and material arrays
    geom* geoms = new geom[renderScene->objects.size()];
    material* materials = new material[renderScene->materials.size()];

    for (int i=0; i < renderScene->objects.size(); i++) 
	{
      geoms[i] = renderScene->objects[i];
    }
    for (int i=0; i < renderScene->materials.size(); i++) 
	{
      materials[i] = renderScene->materials[i];
    }
	
	
	// execute the kernel
    cudaRaytraceCore(dptr, renderCam, targetFrame, iterations, materials, renderScene->materials.size(), geoms, renderScene->objects.size());//, objs, renderScene->objs.size() );
	cudaEventRecord(stop, 0); 
	cudaEventSynchronize(stop); 

	cudaEventElapsedTime(&time, start, stop); 

	totalTime += time;

	std::cout << "Iterations: " << iterations << " Time: " << time <<  std::endl;

	if (iterations == 25)
	{
		std::cout << "Iterations: " << iterations << " Time: " << time << " Average Time: " << totalTime / iterations << std::endl;
	}
    // unmap buffer object
    cudaGLUnmapBufferObject(pbo);
  } else {

    if (!finishedRender) {
      // output image file
      image outputImage(renderCam->resolution.x, renderCam->resolution.y);

      for (int x=0; x < renderCam->resolution.x; x++) {
        for (int y=0; y < renderCam->resolution.y; y++) {
          int index = x + (y * renderCam->resolution.x);
          outputImage.writePixelRGB(renderCam->resolution.x-1-x,y,renderCam->image[index]);
        }
      }
      
      gammaSettings gamma;
      gamma.applyGamma = true;
      gamma.gamma = 1.0;
      gamma.divisor = 1.0; 
      outputImage.setGammaSettings(gamma);
      string filename = renderCam->imageName;
      string s;
      stringstream out;
      out << targetFrame;
      s = out.str();
      utilityCore::replaceString(filename, ".bmp", "."+s+".bmp");
      utilityCore::replaceString(filename, ".png", "."+s+".png");
      outputImage.saveImageRGB(filename);
      cout << "Saved frame " << s << " to " << filename << endl;
      finishedRender = true;
      if (singleFrameMode==true) {
        cudaDeviceReset(); 
        exit(0);
      }
    }
    if (targetFrame < renderCam->frames-1) {

      // clear image buffer and move onto next frame
      targetFrame++;
      iterations = 0;
      for(int i=0; i<renderCam->resolution.x*renderCam->resolution.y; i++){
        renderCam->image[i] = glm::vec3(0,0,0);
      }
      cudaDeviceReset(); 
      finishedRender = false;
    }
  }
}

void clearImage(camera* camera)
{
	int index = camera->resolution.x * camera->resolution.y;
	for (int i = 0; i < index; ++i)
	{
		camera->image[i] = glm::vec3(0.0f);
	}
}

//-------------------------------
//----------SETUP STUFF----------
//-------------------------------

bool init(int argc, char* argv[]) {

	glfwSetErrorCallback(errorCallback);

  if (!glfwInit()) {
      return false;
  }

  width = 800;
  height = 800;
  window = glfwCreateWindow(width, height, "CIS 565 Pathtracer", NULL, NULL);
  if (!window){
      glfwTerminate();
      return false;
  }
  glfwMakeContextCurrent(window);
  glfwSetKeyCallback(window, keyCallback);
  glfwSetMouseButtonCallback(window, mouseCallback);
 
  // Set up GL context
  glewExperimental = GL_TRUE;
  if(glewInit()!=GLEW_OK){
    return false;
  }

  // Initialize other stuff
  initVAO();
  initTextures();
  initCuda();
  initPBO();
  
  GLuint passthroughProgram;
  passthroughProgram = initShader();

  glUseProgram(passthroughProgram);
  glActiveTexture(GL_TEXTURE0);
 
  return true;
}

void initPBO(){
  // set up vertex data parameter
  int num_texels = width*height;
  int num_values = num_texels * 4;
  int size_tex_data = sizeof(GLubyte) * num_values;
    
  // Generate a buffer ID called a PBO (Pixel Buffer Object)
  glGenBuffers(1, &pbo);

  // Make this the current UNPACK buffer (OpenGL is state-based)
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

  // Allocate data for the buffer. 4-channel 8-bit image
  glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
  cudaGLRegisterBufferObject(pbo);

}

void initCuda(){
  // Use device with highest Gflops/s
  cudaGLSetGLDevice(0);

  // Clean up on program exit
  atexit(cleanupCuda);
}

void initTextures(){
    glGenTextures(1, &displayImage);
    glBindTexture(GL_TEXTURE_2D, displayImage);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
}

void initVAO(void){
    GLfloat vertices[] =
    { 
        -1.0f, -1.0f, 
         1.0f, -1.0f, 
         1.0f,  1.0f, 
        -1.0f,  1.0f, 
    };

    GLfloat texcoords[] = 
    { 
        1.0f, 1.0f,
        0.0f, 1.0f,
        0.0f, 0.0f,
        1.0f, 0.0f
    };

    GLushort indices[] = { 0, 1, 3, 3, 1, 2 };

    GLuint vertexBufferObjID[3];
    glGenBuffers(3, vertexBufferObjID);
    
    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)positionLocation, 2, GL_FLOAT, GL_FALSE, 0, 0); 
    glEnableVertexAttribArray(positionLocation);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(texcoords), texcoords, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)texcoordsLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(texcoordsLocation);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertexBufferObjID[2]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

}

GLuint initShader() {
  const char *attribLocations[] = { "Position", "Tex" };
  GLuint program = glslUtility::createDefaultProgram(attribLocations, 2);
  GLint location;
  
  glUseProgram(program);
  if ((location = glGetUniformLocation(program, "u_image")) != -1)
  {
    glUniform1i(location, 0);
  }

  return program;
}

//-------------------------------
//---------CLEANUP STUFF---------
//-------------------------------

void cleanupCuda(){
  if(pbo) deletePBO(&pbo);
  if(displayImage) deleteTexture(&displayImage);
}

void deletePBO(GLuint* pbo){
  if (pbo) {
    // unregister this buffer object with CUDA
    cudaGLUnregisterBufferObject(*pbo);
    
    glBindBuffer(GL_ARRAY_BUFFER, *pbo);
    glDeleteBuffers(1, pbo);
    
    *pbo = (GLuint)NULL;
  }
}

void deleteTexture(GLuint* tex){
    glDeleteTextures(1, tex);
    *tex = (GLuint)NULL;
}
void saveImage()
{
	image outputImage(renderCam->resolution.x, renderCam->resolution.y);

      for (int x=0; x < renderCam->resolution.x; x++) 
	  {
        for (int y=0; y < renderCam->resolution.y; y++) 
		{
			int index = x + (y * renderCam->resolution.x);
			outputImage.writePixelRGB(renderCam->resolution.x-1-x,y,renderCam->image[index]);
        }
      }
      
      gammaSettings gamma;
      gamma.applyGamma = true;
      gamma.gamma = 1.0;
      gamma.divisor = 1.0; 
      outputImage.setGammaSettings(gamma);
      string filename = renderCam->imageName;
      string s;
      stringstream out;
      out << targetFrame;
      s = out.str();
      utilityCore::replaceString(filename, ".bmp", "."+s+".bmp");
      outputImage.saveImageRGB(filename);
      cout << "Saved frame " << s << " to " << filename << endl;
}
//------------------------------
//-------GLFW CALLBACKS---------
//------------------------------

void errorCallback(int error, const char* description){
    fputs(description, stderr);
}

void mouseCallback(GLFWwindow* window, int key, int action, int mods)
{
	if (key == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS)
	{
		double *mouseXPos, *mouseYPos;
		glfwGetCursorPos (window, mouseXPos, mouseYPos); 

		//Convert to world coordinates
		glm::vec3 A = glm::cross(renderCam->views[0], renderCam->ups[0]);
		glm::vec3 B = glm::cross(A, renderCam->ups[0]);	//B is in the correct plane
		glm::vec3 M = renderCam->positions[0] + renderCam->ups[0];			    //Midpoint of screen
		glm::vec3 H = glm::normalize(A) * glm::length(renderCam->views[0]) * glm::tan(glm::radians(renderCam->fov.x));
		glm::vec3 V = glm::normalize(B) * glm::length(renderCam->views[0]) * glm::tan(glm::radians(renderCam->fov.y));
	
		float sx = *mouseXPos / ( (float)renderCam->resolution.x - 1.0f);
		float sy = *mouseXPos / ( (float)renderCam->resolution.y - 1.0f);

		glm::vec3 mouseWorldPos = M + (2*sx - 1.0f) * H + (-2.0f * sy + 1.0f) * V;
		
		std::cout << mouseWorldPos.x << std::endl;
	}

}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods){
    if(key == GLFW_KEY_ESCAPE && action == GLFW_PRESS){
        glfwSetWindowShouldClose(window, GL_TRUE);
    }

	//CAMERA MOTION
	if(key == GLFW_KEY_UP && action == GLFW_PRESS)
	{
		renderCam->positions->y += 0.5;
		clearImage(renderCam);
	}
	if(key == GLFW_KEY_DOWN && action == GLFW_PRESS)
	{
		renderCam->positions->y -= 0.5;
		clearImage(renderCam);
	}
	if(key == GLFW_KEY_LEFT && action == GLFW_PRESS)
	{
		renderCam->positions->x -= 0.5;
		clearImage(renderCam);
	}
	if(key == GLFW_KEY_DOWN && action == GLFW_PRESS)
	{
		renderCam->positions->x += 0.5;
		clearImage(renderCam);
	}
	if(key == GLFW_KEY_RIGHT_SHIFT && action == GLFW_PRESS)
	{
		renderCam->positions->z += 0.5;
		clearImage(renderCam);
	}
	if(key == GLFW_KEY_RIGHT_CONTROL && action == GLFW_PRESS)
	{
		renderCam->positions->z -= 0.5;
		clearImage(renderCam);
	}

	//DEPTH OF FIELD MOTION
	if(key == GLFW_KEY_D && action == GLFW_PRESS)
	{
		if(renderCam->focal >= 0)
		{
			renderCam->focal += 1; 
			clearImage(renderCam);
			std::cout << "DEPTH OF FIELD: ON" << std::endl;
			std::cout << "Focal Length ++ 1" << std::endl;
		}
	}
	if(key == GLFW_KEY_F && action == GLFW_PRESS)
	{
		if(renderCam->focal >= 0)
		{
			renderCam->focal -= 1; 
			clearImage(renderCam);
			std::cout << "DEPTH OF FIELD: ON" << std::endl;
			std::cout << "Focal Length -- 1" << std::endl;
		}
	}

	//SAVE IMAGE
	if(key == GLFW_KEY_S && action == GLFW_PRESS)
	{		
		saveImage();
	}
}
