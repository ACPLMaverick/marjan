#include "Bitmap.h"
Bitmap* Bitmap::currentlyLoading;

Bitmap::Bitmap(unsigned int mid) : Texture()
{
	mipID = mid;
	dataSize = 0;
	data = nullptr;
	Loaded = false;
}


Bitmap::~Bitmap()
{
	
}

void Bitmap::Shutdown()
{
	printf("Bitmap: Breaking loading of mip %d\n", mipID);
	CloseReadOperation();

	if (data != nullptr)
	{
		printf("Bitmap: Freeing resources of mip %d\n", mipID);
		delete[] data;
	}
}

bool Bitmap::Initialize(const string* filePath)
{
	DWORD errorCode = LoadAsync(filePath);
	return true;
}

DWORD Bitmap::LoadAsync(const string* filePath)
{
	hFile = INVALID_HANDLE_VALUE;

	hFile = CreateFile(
		filePath->c_str(),
		GENERIC_READ,
		0,	// dwShareMode = 0 - no file sharing!!!
		NULL,
		OPEN_EXISTING,
		FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED,
		NULL);

	if (hFile == INVALID_HANDLE_VALUE)
	{
		DWORD error = CloseReadOperation();
		if (error != 0)
			return error;
		else
			return GetLastError();
	}

	memset(&asyncHandler, 0, sizeof(OVERLAPPED));

	asyncHandler.Offset = 0;
	asyncHandler.OffsetHigh = 0;

	LPCSTR taskName = TEXT(("ReadEvent for mip " + to_string(m_ID)).c_str());
	HANDLE readEvent = CreateEvent(
		NULL,
		TRUE,
		FALSE,
		taskName);

	if (readEvent == NULL)
	{
		DWORD error = CloseReadOperation();
		if (error != 0)
			return error;
		else
			return GetLastError();
	}

	asyncHandler.hEvent = readEvent;

	dataSize = GetFileSize(hFile, NULL);
	LPDWORD bytesRead = 0;
	
	/////////////////////////////////
	data = new char[dataSize];
	/////////////////////////////////

	BOOL readAsyncResult = ReadFile(
		hFile,
		(LPVOID)data,
		dataSize,
		bytesRead,
		&asyncHandler);

	if (readAsyncResult == FALSE && GetLastError() != ERROR_IO_PENDING)
	{
		DWORD error = CloseReadOperation();
		if (error != 0)
			return error;
		else
			return GetLastError();
	}

	/*DWORD readEventResult = WaitForSingleObject(readEvent, INFINITE);

	if (readEventResult != WAIT_OBJECT_0)
	{
		DWORD error = CloseReadOperation();
		if (error != 0) 
			return error;
		else
			return GetLastError();
	}*/
	
	this->Loaded = false;
	Bitmap::currentlyLoading = this;
	printf("Bitmap: Loading asynchronyously mip %d\n", Bitmap::currentlyLoading->GetMipID());

	return ERROR_SUCCESS;
}

DWORD Bitmap::CloseReadOperation()
{
	BOOL closeEventResult;

	if (asyncHandler.hEvent != NULL)
	{
		closeEventResult = CloseHandle(asyncHandler.hEvent);
		if (closeEventResult == FALSE)
		{
			return GetLastError();
		}
	}

	if (hFile != NULL)
	{
		closeEventResult = CloseHandle(hFile);

		if (closeEventResult == FALSE)
		{
			return GetLastError();
		}
	}

	return ERROR_SUCCESS;
}

unsigned int Bitmap::GetMipID()
{
	return mipID;
}

void Bitmap::LoadedCallback()
{
	Bitmap::currentlyLoading->Loaded = true;
	printf("Bitmap: FINISHED Loading asynchronyously mip %d\n", Bitmap::currentlyLoading->GetMipID());
	
	// tu trzeba zrobiæ z danych teksturê
	int offset = 0x0, width = 0x0, height = 0x0, totalSize = 0x0;
	int h10 = (int)Bitmap::currentlyLoading->data[10];
	int h11 = (int)Bitmap::currentlyLoading->data[11];
	int h12 = (int)Bitmap::currentlyLoading->data[12];
	int h13 = (int)Bitmap::currentlyLoading->data[13];
	offset = ((h13 << 24) + (h12 << 16) + (h11 << 8) + h10);

	width = (((int)Bitmap::currentlyLoading->data[21] << 24) +
		((int)Bitmap::currentlyLoading->data[20] << 16) +
		((int)Bitmap::currentlyLoading->data[19] << 8) +
		((int)Bitmap::currentlyLoading->data[18]));

	height = (((int)Bitmap::currentlyLoading->data[25] << 24) +
		((int)Bitmap::currentlyLoading->data[24] << 16) +
		((int)Bitmap::currentlyLoading->data[23] << 8) +
		((int)Bitmap::currentlyLoading->data[22]));

	totalSize = width * height * FIXED_CHANNELS;

	glTexImage2D(
		GL_TEXTURE_2D,
		0,
		GL_RGB,
		width,
		height,
		0,
		GL_BGRA,
		GL_UNSIGNED_BYTE,
		Bitmap::currentlyLoading->data + offset);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

	//glGenTextures(1, &(Bitmap::currentlyLoading->m_ID));

	//glBindTexture(GL_TEXTURE_2D, Bitmap::currentlyLoading->m_ID);

	TextureManager::TextureLoadedCallback(Bitmap::currentlyLoading);
	Bitmap::currentlyLoading = nullptr;
}

bool Bitmap::CheckIfLoaded()
{
	DWORD readEventResult = WaitForSingleObject(asyncHandler.hEvent, 1);

	if (readEventResult == WAIT_OBJECT_0)
	{
		Bitmap::LoadedCallback();
		return true;
	}
	else if (readEventResult == WAIT_TIMEOUT)
	{
		return false;
	}
	else
	{
		printf("Bitmap: Loading has failed");
		return false;
	}
}
