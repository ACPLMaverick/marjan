#pragma once
#include <glut.h>
#include "Texture2D.h"
#include "Float2.h"

class Sprite
{
private:
	int frameCount;
	int startFrame;
	int endFrame;
	float currentFrame;
	enum state;
	float framesPerSecond;

	Texture2D *myTexture;
public:
	Sprite(Texture2D *texture, int frameCount, int framesPerSecond);
	~Sprite();

	void update();
	void draw(Float2 position, float orientation);
	void play();
	void resume();
	void pause();
};

