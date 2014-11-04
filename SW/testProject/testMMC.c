/******************************************************************************
 *
 * Copyright:
 *    (C) 2000 - 2005 Embedded Artists AB
 *
 *****************************************************************************/


#include <stdio.h>
#include <efs.h>
#include <ls.h>
#include <string.h>
#include <lpc2xxx.h>
#include <consol.h>
#include "mp3shared.h"

#define JOYSTICK_RIGHT 18
#define JOYSTICK_GND 16

SongInfo currentSongInfo;
unsigned char mmcInitialized;
unsigned char changeLeft;
unsigned char changeRight;
unsigned char rewindForward;
unsigned char rewindBackward;
unsigned int isError;
char* error;

/////////////////////////////
unsigned char id3TagSize = 128;
				unsigned long fileSize;
				unsigned char readSize = 12;
				unsigned char myName[13];
				unsigned char myAuthor[13];
				unsigned long titleOffset = 3;
				unsigned long authorOffset = 33;
//////////////////////////////

EmbeddedFileSystem  efs;
EmbeddedFile        file;
DirList             list;
unsigned char       file_name[13];
unsigned int        size;

unsigned char files[256][13];

void format_file_name(unsigned char *dest, unsigned char *src)
{
	unsigned char i, k;

	for (i = 7; i>0 && (src[i] == 0x20); i--);

	strncpy(dest, src, i + 1);
	dest[i + 1] = '.';

	for (k = 2; k>0 && (src[8 + k] == 0x20); k--);

	strncpy(&dest[i + 2], &src[8], k + 1);
	dest[i + 5] = '\0';
}

char* initMMC(void)
{
		char* error01 = "ERR:filesys-1";
		char* error01b = "ERR:filesys-2";
		char* error01c = "ERR:filesys-3";
		char* error01d = "ERR:filesys-4";

		char* error010 = "ERR:filesys-10";
		char* error011 = "ERR:filesys-11";
		char* error012 = "ERR:filesys-12";
		char* error013 = "ERR:filesys-13";
		char* error014 = "ERR:filesys-14";
		char* error015 = "ERR:filesys-15";
		char* error016 = "ERR:filesys-16";
		char* error017 = "ERR:filesys-17";

		char* error02 = "ERR:dir";
		signed int returnValue;
		//initialize SPI?
		returnValue = efs_init(&efs, "\\");

		if (returnValue == -1)
		{
			DBG((TXT("Could not open filesystem.\n")));
			return error01;
		}
		if (returnValue == -2)
			{
				DBG((TXT("Could not open filesystem.\n")));
				return error01b;
			}
		if (returnValue == -3)
			{
				DBG((TXT("Could not open filesystem.\n")));
				return error01c;
			}
		if (returnValue == -4)
			{
				DBG((TXT("Could not open filesystem.\n")));
				return error01d;
			}

		if (returnValue == -10)
				{
					DBG((TXT("Could not open filesystem.\n")));
					return error010;
				}
		if (returnValue == -11)
				{
					DBG((TXT("Could not open filesystem.\n")));
					return error011;
				}
		if (returnValue == -12)
				{
					DBG((TXT("Could not open filesystem.\n")));
					return error012;
				}
		if (returnValue == -13)
				{
					DBG((TXT("Could not open filesystem.\n")));
					return error013;
				}
		if (returnValue == -14)
				{
					DBG((TXT("Could not open filesystem.\n")));
					return error014;
				}
		if (returnValue == -15)
				{
					DBG((TXT("Could not open filesystem.\n")));
					return error015;
				}
		if (returnValue == -16)
				{
					DBG((TXT("Could not open filesystem.\n")));
					return error016;
				}
		if (returnValue == -17)
				{
					DBG((TXT("Could not open filesystem.\n")));
					return error017;
				}

		if (ls_openDir(&list, &(efs.myFs), "/") != 0)
		{
			DBG((TXT("Could not open the selected directory.\n")));
			return error02;
		}
	mmcInitialized = 1;
	return "OK";
}

void getFileNames(void)
{
//		if(mmcInitialized == 0)
//		{
//			char* toReturn = initMMC();
//			if(!(toReturn[0] == 'O' && toReturn[1] == 'K')) return;
//		}

		//unsigned char       file_name[13];
		int i = 0;
		while (ls_getNext(&list) == 0)
		{
			if ((list.currentEntry.FileName[8] == 'M') &&
				(list.currentEntry.FileName[9] == 'P') &&
				(list.currentEntry.FileName[10] == '3'))
			{
				DBG((TXT("Filename: %.11s (%li bytes)\n"), list.currentEntry.FileName, list.currentEntry.FileSize));

				format_file_name(file_name, list.currentEntry.FileName);
				int j;
				for(j = 0; j < 13; j++) files[i][j] = file_name[j];
				i++;
			}
		}

}

void testMMC(char* name)
{
	unsigned char* error03 = "ERR:file";
			if (file_fopen(&file, &efs.myFs, name, 'r') == 0)
			{
				// stream data to playback?
				DBG((TXT("File successfully opened!.\n")));
				int i;
				for(i = 0; i < readSize; i++)
				{
					myName[i] = '\0';
					myAuthor[i] = '\0';
				}
				myName[12] = '\0';
				myAuthor[12] = '\0';
				fileSize = file.FileSize;
				file_fread(&file,(fileSize - id3TagSize) + titleOffset, readSize,myName);
				file_fread(&file,(fileSize - id3TagSize) +  authorOffset, readSize, myAuthor);

				currentSongInfo.name = myName;
				currentSongInfo.author = myAuthor;
				file_fclose(&file);
			}
			else
			{
				DBG((TXT("Could not open file.\n")));
				currentSongInfo.name = error03;
			}
	//fs_umount(&(efs.myFs));
}

void MMCproc(void)
{
			char* toReturn;
			toReturn = initMMC();
			mmcInitialized = 1;
			if(!(toReturn[0] == 'O' && toReturn[1] == 'K'))
			{
				isError = 1;
				error = toReturn;
				mmcInitialized = 0;
			}
	int i;
	for(i = 0; i < 99; i++)
	{
		files[i][0] = "";
	}
	getFileNames();
	testMMC(&files[currentSongInfo.ID][0]);

	IODIR0 &= ~(1<<JOYSTICK_RIGHT);
	IOCLR0 |= (1<<JOYSTICK_RIGHT);
	currentSongInfo.name = "test1";
	while(1)
	{
		if(changeLeft == 1)
		{
			changeLeft = 0;
		}
		if(/*(IOPIN0 & (1<<JOYSTICK_RIGHT)) == 0*/1)
		{
			currentSongInfo.name = "test2";
			changeRight = 0;
		}
		if(rewindForward == 1)
		{
			rewindForward = 0;
		}
		if(rewindBackward == 1)
		{
			rewindBackward = 0;
		}
	}
}
