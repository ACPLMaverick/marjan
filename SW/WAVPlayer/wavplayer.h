/*
 * wavplayer.h
 *
 *  Created on: 2014-12-14
 *      Author: embedded
 */

#ifndef WAVPLAYER_H_
#define WAVPLAYER_H_

#define S_SIZE 15908
#define S_TYPE char
#define S_SAMPLE 600
#define S_START 44

#include <lpc2xxx.h>
#include <consol.h>
#include "./pre_emptive_os/api/general.h"
#include "filesys/efs.h"
#include "mp3shared.h"

void playWAV(EmbeddedFile *file);

#endif /* WAVPLAYER_H_ */
