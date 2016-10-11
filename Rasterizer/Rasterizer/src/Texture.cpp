#include "Texture.h"

#include <fstream>
#include <string>

Texture::Texture()
{
}

Texture::Texture(const std::string * name)
{
	LoadFromFile(name);
}

Texture::~Texture()
{
	if (_data != nullptr)
	{
		delete[] _data;
	}
}

Color32 Texture::GetColor(const math::Float2 * uv, WrapMode wrp, FilterMode fm)
{
	math::Float2 newUV = *uv;
	newUV.v = 1.0f - newUV.v;
	int32_t iu, iv;

	if (wrp == WrapMode::CLAMP)
	{
		newUV.u = Clamp(newUV.u, 0.0f, 1.0f);
		newUV.v = Clamp(newUV.v, 0.0f, 1.0f);
	}
	else if (wrp == WrapMode::WRAP)
	{
		math::Float2 floors(floor(newUV.u), floor(newUV.v));
		newUV = newUV - floors;
	}

	iu = (int32_t)(newUV.u * (float)_width);
	iv = (int32_t)(newUV.v * (float)_width);

	//if (fm == FilterMode::NEAREST)
	//{
		newUV = math::Float2(round(newUV.u), round(newUV.v));
	//}
	//else if (fm == FilterMode::LINEAR)
	//{

	//}

	return _data[iv * (int32_t)_width + iu];
}

void Texture::LoadFromFile(const std::string * name)
{
	std::ifstream file(TEXTURE_PATH + *name + TEXTURE_EXTENSION, std::ios::in | std::ios::binary);
	uint8_t bpp;

	if (file.is_open())
	{
		// check if it is TGA file
		file.seekg(-18, std::ios_base::end);
		char sigBuffer[18];
		file.read(sigBuffer, 18);
		if (strcmp(sigBuffer, "TRUEVISION-XFILE."))
		{
#ifdef _DEBUG

			std::cout << "Not a TGA file: " << TEXTURE_PATH + *name + TEXTURE_EXTENSION << std::endl;

#endif // _DEBUG

			return;
		}

		// get file size and check if it's square
		uint16_t wh[2];
		file.seekg(12, std::ios_base::beg);
		file.read((char*)wh, 4);
		if (wh[0] != wh[1])
		{
#ifdef _DEBUG

			std::cout << "No square dimensions: " << TEXTURE_PATH + *name + TEXTURE_EXTENSION << std::endl;

#endif // _DEBUG

			return;
		}
		_width = wh[0];

		// check whether it is 24-bit or 32-bit file
		file.read((char*)&bpp, 1);

		// create data buffer and fill with BGRA bytes
		file.seekg(1, std::ios_base::cur);
		uint32_t siz = _width * _width;
		_data = new Color32[siz];

		if (bpp == 32)
		{
			file.read((char*)_data, siz * bpp);
		}
		else if (bpp == 24)
		{
			uint8_t tempBuffer[3];

			for (size_t i = 0; i < siz; ++i)
			{
				file.read((char*)tempBuffer, 3);
				_data[i] = Color32(255, tempBuffer[2], tempBuffer[1], tempBuffer[0]);
			}
		}

		return;
	}

#ifdef _DEBUG

	std::cout << "Unable to open file: " << TEXTURE_PATH + *name + TEXTURE_EXTENSION << std::endl;

#endif // _DEBUG
}