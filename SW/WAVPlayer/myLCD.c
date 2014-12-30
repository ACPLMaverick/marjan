#include "myLCD.h"


unsigned char currentDisplayMode;
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
extern unsigned char displayMode;

/*****************************************************************************
 *
 * Description:
 *    xxx
 *
 ****************************************************************************/
void initLCD(void)
{
	IODIR1 |= (LCD_DATA | LCD_E | LCD_RS);
	IOCLR1  = (LCD_DATA | LCD_E | LCD_RS);

	IODIR0 |= LCD_RW;
	IOCLR0  = LCD_RW;
	
  IODIR0 |= LCD_BACKLIGHT;
  IOCLR0  = LCD_BACKLIGHT;

  currentDisplayMode = displayMode;
}

/*****************************************************************************
 *
 * Description:
 *    xxx
 *
 ****************************************************************************/
void delay37us(void)
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
void writeLCD(tU8 reg, tU8 data)
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
void lcdBacklight(tU8 onOff)
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
unsigned char numberToChar(unsigned long number)
{
	return (number % 10) + 48;
}

/*****************************************************************************
 *
 * Description:
 *    Converts time to format MM:SS as string
 *
 ****************************************************************************/
void timeToString(unsigned long time, unsigned char* toReturn)
{
	unsigned long maxTime = 3599;
	while (time > maxTime) time = maxTime;

	unsigned int minutes = time / 60;
	unsigned int seconds = time % 60;
	toReturn[0] = minutes / 10 + 48;
	toReturn[1] = minutes % 10 + 48;
	toReturn[2] = ':';
	toReturn[3] = seconds / 10 + 48;
	toReturn[4] = seconds % 10 + 48;
	toReturn[5] = '\0';
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

void startLCD(void)
{
  initLCD();
  lcdBacklight(FALSE);
  osSleep(10);
  unsigned char awaitLoop = 1;

  LCDWriteNameAuthor();
//  for(;;)
//  {
// 	clearScr();
//
//// 	if(displayMode == 1)
//// 	{
//// 		//writeLCD(1, numberToChar(currentSongInfo.time));
//// 		unsigned char toWrite[6];
//// 		timeToString(currentSongInfo.time, toWrite);
//// 		WriteString(toWrite);
//// 		osSleep(10);
//// 		continue;
//// 	}
//
// 	if(volumeUp != 0 || volumeDown != 0)
// 	{
// 		writeLCD(1, numberToChar(currentVolume));
// 		if(volumeUp != 0) volumeUp = 0;
// 		if(volumeDown != 0) volumeDown = 0;
// 		awaitLoop = 0;
// 	}
// 	else if(isError == 0)
//	{
//		WriteString(currentSongInfo.name);
//
//		//move curstor to second row
//		writeLCD(0, 0x80 | 0x40);
//		delay37us();
//
//		//WriteString(currentSongInfo.author);
//		unsigned char toWrite[6];
//		timeToString(currentSongInfo.time, toWrite);
//		WriteString(toWrite);
//	}
//	else
//	{
//		WriteString(error);
//	}
//
//    osSleep(50);
//    osSleep(1); //actually only 1.52 mS needed
////
////    // awaiting command from joystick
////    if(awaitLoop != 0)
////    {
////    	for(;;)
////    	    {
////    	    	if(changeLeft != 0)
////    	    	{
////    	    		changeLeft = 0;
////    	    		break;
////    	    	}
////    	    	if(changeRight != 0)
////    	    	{
////    	    	    changeRight = 0;
////    	    	    break;
////    	    	}
////    	    	if(volumeUp != 0)
////    	    	{
////    	    		break;
////    	    	}
////    	    	if(volumeDown != 0)
////    	    	{
////    	    		break;
////    	    	}
////    	    	if(isError != 0)
////    	    	{
////    	    		break;
////    	    	}
////    	    	if(currentDisplayMode != displayMode)
////    	    	{
////    	    		currentDisplayMode = displayMode;
////    	    		break;
////    	    	}
////
////    	    	osSleep(100);
////    	    	if(currentSongInfo.nameLength > 16 || currentSongInfo.authorLength > 16)
////    	    	{
////    	    		ScreenShiftLeft();
////    	    	}
////    	    }
////    }
//    if(awaitLoop == 0) awaitLoop = 1;
//  }
	//lcdBacklight(FALSE);
}

void LCDWriteNameAuthor(void)
{
	clearScr();

	WriteString(currentSongInfo.name);

	//move curstor to second row
	writeLCD(0, 0x80 | 0x40);
	delay37us();

	WriteString(currentSongInfo.author);
}

void LCDWriteVolume(void)
{
	clearScr();

	writeLCD(1, numberToChar(currentVolume));
	if(volumeUp != 0) volumeUp = 0;
	if(volumeDown != 0) volumeDown = 0;
}

void LCDWriteTime(void)
{
	clearScr();

	unsigned char toWrite[6];
	timeToString(currentSongInfo.time, toWrite);
	WriteString(toWrite);
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
