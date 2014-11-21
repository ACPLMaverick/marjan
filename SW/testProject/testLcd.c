/******************************************************************************
 *
 * Copyright:
 *    (C) 2000 - 2005 Embedded Artists AB
 *
 *****************************************************************************/


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

/////////////////////////////
extern SongInfo currentSongInfo;
extern unsigned char mmcInitialized;
extern unsigned char changeLeft;
extern unsigned char changeRight;
extern unsigned char rewindForward;
extern unsigned char rewindBackward;
extern unsigned char volumeUp;
extern unsigned char volumeDown;
extern unsigned int isError;
extern char* error;
extern unsigned char currentVolume;
////////////////////////////
void ScreenShiftLeft(void);
////////////////////////////

/*****************************************************************************
 *
 * Description:
 *    xxx
 *
 ****************************************************************************/
static void
initLCD(void)
{
	IODIR1 |= (LCD_DATA | LCD_E | LCD_RS);
	IOCLR1  = (LCD_DATA | LCD_E | LCD_RS);

	IODIR0 |= LCD_RW;
	IOCLR0  = LCD_RW;
	
  IODIR0 |= LCD_BACKLIGHT;
  IOCLR0  = LCD_BACKLIGHT;
}

/*****************************************************************************
 *
 * Description:
 *    xxx
 *
 ****************************************************************************/
static void
delay37us(void)
{
	volatile tU32 i;

//Temp test for 140 uS delay
	for(i=0; i<6*2500; i++)
    asm volatile (" nop"); //delay 15 ns x 2500 = about 37 us delay
}

/*****************************************************************************
 *
 * Description:
 *    xxx
 *
 ****************************************************************************/
static void
writeLCD(tU8 reg, tU8 data)
{
	volatile tU8 i;

	if (reg == 0)
	  IOCLR1 = LCD_RS;
	else
	  IOSET1 = LCD_RS;
	  
  IOCLR0 = LCD_RW;
	IOCLR1 = LCD_DATA;
	IOSET1 = ((tU32)data << 16) & LCD_DATA;
//printf("\nwrite: %x", data);
	
	IOSET1 = LCD_E;
//	osSleep(1);
	for(i=0; i<16; i++)
    asm volatile (" nop"); //delay 15 ns x 16 = about 250 ns delay
	IOCLR1 = LCD_E;
//	osSleep(1);
	for(i=0; i<16; i++)
    asm volatile (" nop"); //delay 15 ns x 16 = about 250 ns delay
}

/*****************************************************************************
 *
 * Description:
 *    xxx
 *
 ****************************************************************************/
static void
lcdBacklight(tU8 onOff)
{
	if (onOff == TRUE)
	  IOSET0 = LCD_BACKLIGHT;
	else
	  IOCLR0 = LCD_BACKLIGHT;
}

/*****************************************************************************
 *
 * Description:
 *    Returns one-digit number as char
 *    Bigger numbers are "cut" by modulo operation
 *
 ****************************************************************************/
unsigned char numberToChar(unsigned char number)
{
	return (number % 10) + 48;
}

/*****************************************************************************
 *
 * Description:
 *    Converts time to format MM:SS as string
 *
 ****************************************************************************/
unsigned char* timeToString(unsigned long time)
{
	unsigned char str[6];
	unsigned long maxTime = 3599;
	while(time > maxTime) time -= maxTime;
	int i = 0;
	int b = 1000;
	for(i = 0; i < 6; i++, b/10)
	{
		if(i == 2)
		{
			str[i] = ':';
			continue;
		}
		else if(i == 5)
		{
			str[i] = '\0';
			break;
		}
		else
		{
			str[i] = (time / b) + 48;
			time = time - (time/b)*b;
		}
	}

	return str;
}

void clearScr()
{
	lcdBacklight(TRUE);
    osSleep(50);

    //function set
    writeLCD(0, LCD_FC_8BIT_1LINE_5x7);
    osSleep(1);
    writeLCD(0, LCD_FC_8BIT_1LINE_5x7);
    delay37us();
    writeLCD(0, LCD_FC_8BIT_1LINE_5x7);
    delay37us();

    //function set
    writeLCD(0, LCD_FC_8BIT_2LINE_5x7);
    delay37us();

    //display off
    writeLCD(0, LCD_CURSOR_OFF_NOCLEAR);
    delay37us();

    //display clear
    writeLCD(0, LCD_DISPLAY_CLEAR);
    osSleep(1); //actually only 1.52 mS needed
      
    //display control set
    writeLCD(0, LCD_ENTRY_MODE);
    osSleep(1);

    //display control set
    writeLCD(0, LCD_CURSOR_OFF);
    delay37us();

    //cursor home
    writeLCD(0, LCD_CURSOR_HOME);
    osSleep(1);
}

void
testLcd(void)
{
  initLCD();
  lcdBacklight(FALSE);
  osSleep(10);
  unsigned char awaitLoop = 1;
  for(;;)
  {
 	clearScr();

 	/// TIMER TEST!!!!!!
 	{
 		//writeLCD(1, numberToChar(currentSongInfo.time));
 		WriteString(timeToString(currentSongInfo.time));
 		osSleep(10);
 		continue;
 	}

 	if(volumeUp != 0 || volumeDown != 0)
 	{
 		writeLCD(1, numberToChar(currentVolume));
 		if(volumeUp != 0) volumeUp = 0;
 		if(volumeDown != 0) volumeDown = 0;
 		awaitLoop = 0;
 	}
 	else if(isError == 0)
	{
		WriteString(currentSongInfo.name);

		//move curstor to second row
		writeLCD(0, 0x80 | 0x40);
		delay37us();

		WriteString(currentSongInfo.author);
	}
	else
	{
		WriteString(error);
	}

    osSleep(50);
    osSleep(1); //actually only 1.52 mS needed

    // awaiting command from joystick
    if(awaitLoop != 0)
    {
    	for(;;)
    	    {
    	    	if(changeLeft != 0)
    	    	{
    	    		changeLeft = 0;
    	    		break;
    	    	}
    	    	if(changeRight != 0)
    	    	{
    	    	    changeRight = 0;
    	    	    break;
    	    	}
    	    	if(volumeUp != 0)
    	    	{
    	    		break;
    	    	}
    	    	if(volumeDown != 0)
    	    	{
    	    		break;
    	    	}
    	    	if(isError != 0)
    	    	{
    	    		break;
    	    	}

    	    	osSleep(100);
    	    	if(currentSongInfo.nameLength > 16 || currentSongInfo.authorLength > 16)
    	    	{
    	    		ScreenShiftLeft();
    	    	}
    	    }
    }
    if(awaitLoop == 0) awaitLoop = 1;
  }
	//lcdBacklight(FALSE);
}

void WriteString(const char* str)
{
	int i = 0;
	for(; str[i] != '\0'; i++)
	{
		writeLCD(1, str[i]);
		delay37us();
	}
}


void ScreenShiftLeft(void)
{
	writeLCD(0, LCD_FC_8BIT_2LINE_5x7);
	delay37us();
	writeLCD(0, LCD_SHIFT_LEFT);
	osSleep(1);
	writeLCD(0, LCD_ENTRY_MODE);
	osSleep(1);
}
