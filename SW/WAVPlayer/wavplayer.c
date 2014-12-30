/*
 * wavplayer.c
 *
 *  Created on: 2014-12-14
 *      Author: embedded
 */

#include "wavplayer.h"
#include "myLCD.h"
#include "myMMC.h"
#include "myTimer.h"

SongInfo currentSongInfo;
EmbeddedFile* myFile;
S_TYPE soundTable[S_SIZE];
unsigned long* filePtr;
unsigned long fileOffset;
unsigned long fileSize;
unsigned long joystickTimer;

unsigned char currentVolume;

void zeroBuffer(void)
{
	int i;
	for(i = 0; i < S_SIZE; i++)
	{
		if(i % 2 != 0) soundTable[i] = 255;
		else soundTable[i] = 0;
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
	            (1 << 16);         //BIAS = 1, 2.5uS settling time
}

void playWAV(EmbeddedFile* file)
{
	// reading necessary data from file
	currentSongInfo.time = 0;
	fileSize = (*file).FileSize;
	(*file).FilePtr = S_START;
	fileOffset = S_START;
	myFile = file;
	joystickTimer = 0;

	myTimerExec();
}

void ISR(void)
{
	unsigned char newSample[1];
	file_read(myFile, 1, newSample);
	fileOffset += S_SIZE;
	sendToDAC(newSample[0]);

	// koniec piosenki
	if(fileOffset > (fileSize - ID3TAGSIZE))
	{
		StopInterrupts();
		changeRight = 1;
		return;
	}

	// joystick w lewo
	if((IOPIN0 & (1<<JOYSTICK_LEFT)) == 0)
	{
		joystickTimer++;
		if(joystickTimer > (T_CHECKBUTTON/(PRESCALE*DELAY_MS)))
		{
			StopInterrupts();
			joystickTimer = 0;
			changeLeft = 1;
		}
		return;
	}
	//else joystickTimer = 0;

	// joystick w prawo
	if((IOPIN0 & (1<<JOYSTICK_RIGHT)) == 0)
	{
		joystickTimer++;
		if(joystickTimer > (T_CHECKBUTTON/(PRESCALE*DELAY_MS)))
		{
			StopInterrupts();
			joystickTimer = 0;
			changeRight = 1;
		}
		return;
	}
	//else joystickTimer = 0;

	// joystick w gore
	if((IOPIN0 & (1<<JOYSTICK_UP)) == 0)
	{
		joystickTimer++;
		if(joystickTimer > (T_CHECKBUTTON/(PRESCALE*DELAY_MS)))
		{
			joystickTimer = 0;
			volumeUp = 1;
		}
		return;
	}
	//else joystickTimer = 0;

	// joystick w dol
	if((IOPIN0 & (1<<JOYSTICK_DOWN)) == 0)
	{
		joystickTimer++;
		if(joystickTimer > (T_CHECKBUTTON/(PRESCALE*DELAY_MS)))
		{
			joystickTimer = 0;
			volumeDown = 1;
		}
		return;
	}
	//else joystickTimer = 0;

	// joystick wcisniety
	if((IOPIN0 & (1<<JOYSTICK_GND)) == 0)
	{
		joystickTimer++;
		if(joystickTimer > (T_CHECKBUTTON/(PRESCALE*DELAY_MS)))
		{
			joystickTimer = 0;
			displayMode = 1;
		}
		return;
	}
	//else joystickTimer = 0;
}
