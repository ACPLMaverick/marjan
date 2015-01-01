/*
 * wavplayer.h
 *
 *  Created on: 2014-12-14
 *      Author: embedded
 */

#ifndef WAVPLAYER_H_
#define WAVPLAYER_H_

//15908
#define S_SIZE 1
#define S_TYPE unsigned char
#define S_SAMPLE 600
#define S_START 44

#define T_CHECKBUTTON 12000000

#include <lpc2xxx.h>
#include <consol.h>
#include "./pre_emptive_os/api/general.h"
#include "filesys/efs.h"
#include "mp3shared.h"
#include "myTimerRTC.h"

void playWAV(EmbeddedFile* file);
void ISR(void);
void ISR_RTC(void);

#endif /* WAVPLAYER_H_ */
