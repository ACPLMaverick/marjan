/*
 * wavplayer.c
 *
 *  Created on: 2014-12-14
 *      Author: embedded
 */

#include "wavplayer.h"
#include "myLCD.h"
#include "myMMC.h"

SongInfo currentSongInfo;
EmbeddedFile* myFile;
S_TYPE soundTable[S_SIZE];
unsigned long* filePtr;
unsigned long fileOffset;
unsigned long fileSize;
unsigned long joystickTimer;
signed long rewindTimer;
signed long rewTime;
unsigned char canPlay;

unsigned char currentVolume;
unsigned char displayChanged;

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
	rewindTimer = 0;
	rewTime = 0;
	canPlay = 1;

	myTimerRTCExec();
}

void ISR(void)
{
	if(canPlay != 0)
	{
		unsigned char newSample[1];
		file_read(myFile, 1, newSample);
		fileOffset += S_SIZE;
		sendToDAC(newSample[0]);
	}

	// koniec piosenki
	if((*myFile).FilePtr > (fileSize - ID3TAGSIZE))
	{
		StopInterrupts();
		changeRight = 1;
		return;
	}

	// joystick w lewo
	if((IOPIN0 & (1<<JOYSTICK_LEFT)) == 0)
	{
		joystickTimer++;
		rewindTimer--;
		if(joystickTimer > (T_REWDELAY*T_CHECKBUTTON/(PRESCALE*DELAY_MS)) && (IOPIN0 & (1<<JOYSTICK_LEFT)) == 0)
		{
			// przewijanie
			if((*myFile).FilePtr > S_START)
			{
				rewindBackward = 1;
				canPlay = 0;
				rewindTimer = 0;
				displayMode = 2;
				(*myFile).FilePtr -= T_REWIND;
			}
		}
		return;
	}
	// joystick w prawo
	else if((IOPIN0 & (1<<JOYSTICK_RIGHT)) == 0)
	{
		joystickTimer++;
		rewindTimer++;
		if(joystickTimer > (T_REWDELAY*T_CHECKBUTTON/(PRESCALE*DELAY_MS)) && (IOPIN0 & (1<<JOYSTICK_RIGHT)) == 0)
		{
			// przewijanie
			if((*myFile).FilePtr < (*myFile).FileSize)
			{
				rewindForward = 1;
				canPlay = 0;
				rewindTimer = 0;
				displayMode = 2;
				(*myFile).FilePtr += T_REWIND;
			}
		}
		return;
	}
	// joystick w gore
	else if((IOPIN0 & (1<<JOYSTICK_UP)) == 0)
	{
		joystickTimer++;
		if(joystickTimer > (T_CHECKBUTTON/(PRESCALE*DELAY_MS)) && (IOPIN0 & (1<<JOYSTICK_UP)) == 0)
		{
			joystickTimer = 0;
			volumeUp = 1;
		}
		return;
	}
	// joystick w dol
	else if((IOPIN0 & (1<<JOYSTICK_DOWN)) == 0)
	{
		joystickTimer++;
		if(joystickTimer > (T_CHECKBUTTON/(PRESCALE*DELAY_MS)) && (IOPIN0 & (1<<JOYSTICK_DOWN)) == 0)
		{
			joystickTimer = 0;
			volumeDown = 1;
		}
		return;
	}
	// joystick wcisniety
	else if((IOPIN0 & (1<<JOYSTICK_GND)) == 0)
	{
		joystickTimer++;
		if(joystickTimer > (T_CHECKBUTTON/(PRESCALE*DELAY_MS)) && (IOPIN0 & (1<<JOYSTICK_GND)) == 0)
		{
			joystickTimer = 0;
			if(displayMode == 0)
			{
				displayMode = 2;
				return;
			}
//			if(displayMode == 2 || displayMode == 3)
//			{
//				displayMode = 0;
//				return;
//			}
		}
		return;
	}
	else if(rewindTimer > (T_CHECKBUTTON/(PRESCALE*DELAY_MS)))
	{
		StopInterrupts();
		joystickTimer = 0;
		rewindTimer = 0;
		changeRight = 1;
		return;
	}
	else if(rewindTimer < -(T_CHECKBUTTON/(PRESCALE*DELAY_MS)))
	{
		StopInterrupts();
		joystickTimer = 0;
		rewindTimer = 0;
		changeLeft = 1;
		return;
	}
	if(canPlay == 0)
	{
		canPlay = 1;
		displayMode = 0;
	}
	rewindForward = 0;
	rewindBackward = 0;
	joystickTimer = 0;
}

void ISR_RTC(void)
{
	if(canPlay == 0)
	{
		if(rewindForward == 1) rewTime += T_REWIND*0.8;
		if(rewindBackward == 1) rewTime -= T_REWIND*0.8;
	}
	currentSongInfo.time = RTC_SEC + 60*RTC_MIN + rewTime;
	if(displayMode == 2)
	{
		displayMode = 3;
		return;
	}
	if(displayMode == 3)
	{
		displayMode = 2;
		return;
	}
}
