#include "Sprite.h"


Sprite::Sprite(Texture2D *texture, int frameCount, int framesPerSecond)
{
	this->myTexture = texture;
	this->frameCount = frameCount;
	this->framesPerSecond = framesPerSecond;
}


Sprite::~Sprite()
{
	delete myTexture;
}

void Sprite::update()
{

}
void Sprite::draw(Float2 position, float orientation)
{

}

void Sprite::play()
{

}

void Sprite::resume()
{

}

void Sprite::pause()
{

}
