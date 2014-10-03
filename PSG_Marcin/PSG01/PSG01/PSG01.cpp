// PSG01.cpp : Defines the entry point for the console application.
//

#define _USE_MATH_DEFINES
#include <iostream>
#include <cmath>
#include <glut.h>

#include "Texture2D.h"
#include "Sprite.h"
#include "ResourceFactory.h"

using namespace std;

const int width = 800;
const int height = 600;

ResourceFactory *myFactory = new ResourceFactory();

////////////////////////////
void Init(void);
void Reshape(int w, int h);
void Draw(void);
void Idle(void);

int main(int argc, char* argv[])
{
	glutInit(&__argc, __argv);
	glutInitWindowSize(width, height);
	glutInitWindowPosition(100, 100);

	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
	glutCreateWindow("Lubie domki");
	Init();
	glutDisplayFunc(Draw);
	glutReshapeFunc(Reshape);
	glutIdleFunc(Idle);
	//glutKeyboardFunc(PressKey);
	//glutSpecialFunc(PressSpecialKey);
	//glutKeyboardUpFunc(ReleaseKey);
	glutIgnoreKeyRepeat(1);
	//glutPassiveMotionFunc(MouseFunc);

	//CreateMenus();

	glutMainLoop();

	delete myFactory;
	return 0;
}

void Init(void)
{
	glClearColor(0,0,0,1);
	glShadeModel(GL_SMOOTH);
	glEnable(GL_DEPTH_TEST);
	//glEnable(GL_CULL_FACE);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_LIGHTING);
	glEnable(GL_NORMALIZE);
	glEnable(GL_TEXTURE_2D);

	/////////////////////////////////

	
}

void Reshape(int w, int h)
{
	glViewport(0, 0, (GLsizei)w, (GLsizei)h);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	if (w <= h)
	{
		gluOrtho2D(-1.0, 1.0, -1.0*(GLfloat)h / (GLfloat)w, 1.0*(GLfloat)h / (GLfloat)w);
	}
	else
	{
		gluOrtho2D(-1.0*(GLfloat)h / (GLfloat)w, 1.0*(GLfloat)h / (GLfloat)w, -1.0, 1.0);
	}
	glutPostRedisplay();
}

void Draw(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glMatrixMode(GL_MODELVIEW);

	glFlush();
	glutSwapBuffers();
}

void Idle(void)
{
	glutPostRedisplay();
}

