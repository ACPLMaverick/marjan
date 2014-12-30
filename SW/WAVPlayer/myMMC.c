#include "myMMC.h"
#include <stdio.h>
#include "./pre_emptive_os/api/general.h"
#include <efs.h>
#include <ls.h>
#include <string.h>
#include <lpc2xxx.h>
#include <consol.h>

SongInfo currentSongInfo;
unsigned char mmcInitialized;
unsigned char changeLeft;
unsigned char changeRight;
unsigned char rewindForward;
unsigned char rewindBackward;
unsigned char volumeUp;
unsigned char volumeDown;
unsigned int isError;
unsigned char displayMode;
char* error;

/////////////////////////////
unsigned char myName[30];
unsigned char myAuthor[30];
//////////////////////////////

EmbeddedFileSystem  efs;
EmbeddedFile        file;
DirList             list;
unsigned char       file_name[13];
unsigned int        size;

unsigned char songCount;

extern void playWAV(EmbeddedFile *file);

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
		int i = 0;
		while (ls_getNext(&list) == 0)
		{
			if ((list.currentEntry.FileName[8] == 'W') &&
				(list.currentEntry.FileName[9] == 'A') &&
				(list.currentEntry.FileName[10] == 'V'))
			{
				DBG((TXT("Filename: %.11s (%li bytes)\n"), list.currentEntry.FileName, list.currentEntry.FileSize));

				format_file_name(file_name, list.currentEntry.FileName);
				int j;
				for(j = 0; j < 13; j++) files[i][j] = file_name[j];
				i++;
			}
		}
		songCount = i;

}

void ReadTagsFromFile(EmbeddedFile* myFile)
{
	int i;
	for(i = 0; i < READSIZE; i++)
	{
		myName[i] = '\0';
		myAuthor[i] = '\0';
	}

	file_fread(myFile,((*myFile).FileSize - ID3TAGSIZE) + TITLEOFFSET, READSIZE,myName);
	file_fread(myFile,((*myFile).FileSize - ID3TAGSIZE) +  AUTHOROFFSET, READSIZE, myAuthor);

	myName[READSIZE - 1] = '\0';
	myAuthor[READSIZE - 1] = '\0';

	currentSongInfo.name = myName;
	currentSongInfo.author = myAuthor;
	for(i = 0; myName[i] != '\0'; i++);
	currentSongInfo.nameLength = i;
	for(i = 0; myAuthor[i] != '\0'; i++);
	currentSongInfo.authorLength = i;
}

void ReadAndPlay(char* name)
{
	unsigned char* error03 = "ERR:file";
			if (file_fopen(&file, &efs.myFs, name, 'r') == 0)
			{
				DBG((TXT("File successfully opened!.\n")));

				ReadTagsFromFile(&file);

				LCDWriteNameAuthor();

				playWAV(&file);

				//file_fclose(&file);
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

	startLCD();

	ReadAndPlay(&files[currentSongInfo.ID][0]);

	while(1)
	{
		if(changeLeft != 0)
		{
			changeLeft = 0;
			file_fclose(&file);
			currentSongInfo.ID = ((currentSongInfo.ID - 1) % songCount);
			if(currentSongInfo.ID < 0) currentSongInfo.ID = 0;
			currentSongInfo.time = 0;
			ReadAndPlay(&files[currentSongInfo.ID][0]);
		}

		if(changeRight != 0)
		{
			changeRight = 0;
			file_fclose(&file);
			currentSongInfo.ID = ((currentSongInfo.ID + 1) % songCount);
			currentSongInfo.time = 0;
			ReadAndPlay(&files[currentSongInfo.ID][0]);
		}

		if(volumeUp != 0)
		{
			volumeUp = 0;
			if(currentVolume < 9) currentVolume++;
		}

		if(volumeDown != 0)
		{
			volumeDown = 0;
			if(currentVolume > 0) currentVolume--;
		}

		if(rewindForward != 0)
		{

		}

		if(rewindBackward != 0)
		{

		}

		if(displayMode != 0)
		{
			if(displayMode == 0) displayMode = 1;
			else displayMode = 0;
		}
	}
}
