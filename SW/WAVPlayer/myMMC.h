/*
 * myMMC.h
 *
 *  Created on: 2014-12-30
 *      Author: embedded
 */

#ifndef MYMMC_H_
#define MYMMC_H_

#include "mp3shared.h"
#include "wavplayer.h"
#include "myLCD.h"

#define JOYSTICK_UP 17
#define JOYSTICK_RIGHT 18
#define JOYSTICK_LEFT 19
#define JOYSTICK_DOWN 20
#define JOYSTICK_GND 14

#define BUTTONCHECK_DELAY 20
#define BUTTONREWINDCHECK_DELAY 100

#define ID3TAGSIZE 128
#define READSIZE 30
#define TITLEOFFSET 3
#define AUTHOROFFSET 33

unsigned char files[256][13];

void MMCproc(void);
void ReadAndPlay(char* name);

#endif /* MYMMC_H_ */
