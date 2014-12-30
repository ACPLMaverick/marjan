/*
 * wavplayer.c
 *
 *  Created on: 2014-12-14
 *      Author: embedded
 */

#include "wavplayer.h"

SongInfo currentSongInfo;
EmbeddedFile* myFile;
S_TYPE soundTable[S_SIZE];
unsigned long* filePtr;
unsigned long fileOffset;
unsigned long fileSize;

unsigned char currentVolume;

extern void myTimerExec(void);

void zeroBuffer(void)
{
	int i;
	for(i = 0; i < S_SIZE; i++)
	{
		if(i % 2 != 0) soundTable[i] = 255;
		else soundTable[i] = 0;
	}
}

void sendToDACOld(void)
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

void sendToDAC(unsigned char sample)
{
		tS32 val;
		float volumeMultiplier = (float)currentVolume / 10;
		val = sample;
		val -= 128;
	    val = val * 2 * volumeMultiplier;
	    if (val > 127) val = 127;
	    else if (val < -127) val = -127;
	    DACR = ((val+128) << 8) |  //actual value to output
	            (0 << 16);         //BIAS = 1, 2.5uS settling time
}

void readFromFile(void)
{
	//zeroBuffer();
	file_read(myFile, S_SIZE, soundTable);
	//fileOffset = fileOffset + S_SIZE;
	//currentSongInfo.time = fileOffset;
}

void playWAV(EmbeddedFile* file)
{
	// reading necessary data from file
	currentSongInfo.time = 0;
	fileSize = (*file).FileSize;
	(*file).FilePtr = S_START;
	//fileOffset = S_START;
	myFile = file;

	myTimerExec();

	// main function loop
//	while(fileOffset < fileSize)
//	{
//		readFromFile();
//		sendToDACOld();
//	}
//	unsigned long tmp = fileOffset - fileSize;
//	if(tmp < S_SIZE)
//	{
//		fileOffset = fileOffset - (fileOffset - fileSize);
//		readFromFile();
//		sendToDACOld();
//	}
}

void ISR(void)
{
	unsigned char newSample[1];
	file_read(myFile, 1, newSample);
	sendToDAC(newSample[0]);
}
