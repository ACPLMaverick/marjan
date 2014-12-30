/*
 * wavplayer.h
 *
 *  Created on: 2014-12-14
 *      Author: embedded
 */

#ifndef MYLCD_H_
#define MYLCD_H_

#include "../pre_emptive_os/api/osapi.h"
#include "../pre_emptive_os/api/general.h"
#include <printf_P.h>
#include <lpc2xxx.h>
#include <consol.h>
#include "mp3shared.h"

#define LCD_DATA      0x00ff0000  //P1.16-P1.23
#define LCD_E         0x02000000  //P1.25
#define LCD_RW        0x00400000  //P0.22
#define LCD_RS        0x01000000  //P1.24
#define LCD_BACKLIGHT 0x40000000  //P0.30

#define LCD_FC_8BIT_1LINE_5x7 0x30
#define LCD_FC_8BIT_2LINE_5x7 0x38
#define LCD_CURSOR_OFF_NOCLEAR 0x08
#define LCD_DISPLAY_CLEAR 0x01
#define LCD_ENTRY_MODE 0x06
#define LCD_CURSOR_OFF 0x0c
#define LCD_CURSOR_HOME 0x02
#define LCD_DISPLAY_OFF 0x80
#define LCD_CURSOR_2ROW 0xC0
#define LCD_SHIFT_LEFT 0x18

void initLCD(void);
void delay37us(void);
void writeLCD(tU8 reg, tU8 data);
void lcdBacklight(tU8 onOff);
unsigned char numberToChar(unsigned long number);
void timeToString(unsigned long time, unsigned char* toReturn);
void clearScr();
void startLCD(void);
void WriteString(const char* str);
void ScreenShiftLeft(void);

void LCDWriteNameAuthor(void);
void LCDWriteVolume(void);
void LCDWriteTime(void);


#endif
