/*
 * wavplayer.c
 *
 *  Created on: 2014-12-14
 *      Author: embedded
 */

#include "wavplayer.h"

EmbeddedFile* myFile;
S_TYPE soundTable[S_SIZE];
unsigned long* filePtr;
unsigned long fileOffset;
unsigned long fileSize;

unsigned char currentVolume;

void sendToDAC(void)
{
	tU32 cnt = 0;
	int i = 0;
	while(cnt++ < S_SIZE)
	{
		tS32 val;
		float volumeMultiplier = (float)currentVolume / 10;
		val = soundTable[cnt] - 128;
	    val = val * 2 * volumeMultiplier;
	    if (val > 127) val = 127;
	    else if (val < -127) val = -127;

	    DACR = ((val+128) << 8) |  //actual value to output
	            (1 << 16);         //BIAS = 1, 2.5uS settling time

	    for(i=0; i < S_SAMPLE; i++)
	      asm volatile (" nop");
	    }
}

void readFromFile(void)
{
	file_fread(myFile, fileOffset, S_SIZE, soundTable);
	fileOffset += S_SIZE;
}

void playWAV(EmbeddedFile *file)
{
	// reading necessary data from file
	fileSize = (*file).FileSize;
	filePtr = &((*file).FilePtr);
	fileOffset = S_START;
	myFile = file;

	// main function loop
	while(fileOffset < fileSize)
	{
		readFromFile();
		sendToDAC();
	}
	unsigned long tmp = fileOffset - fileSize;
	if(tmp < S_SIZE)
	{
		fileOffset = fileOffset - (fileOffset - fileSize);
		readFromFile();
		sendToDAC();
	}
}
