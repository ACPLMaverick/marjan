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

//extern signed char ls_openDir(DirList *dlist,FileSystem *fs,char* dirname);
//extern signed char ls_getNext(DirList *dlist);
//extern signed char ls_getDirEntry(DirList *dlist);
//extern signed char ls_getRealDirEntry(DirList *dlist);
//extern signed char ls_getRootAreaEntry(DirList *dlist);
//extern signed char ls_isValidFileEntry(ListDirEntry *entry);
//extern void   ls_fileEntryToDirListEntry(DirList *dlist, unsigned char* buf, unsigned short offset);
//extern signed int efs_init(EmbeddedFileSystem * efs, char* opts);


EmbeddedFileSystem  efs;
EmbeddedFile        file;
DirList             list;
unsigned char       file_name[13];
unsigned int        size;

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


char* testMMC(void)
{
//	EmbeddedFileSystem  efs;
//	EmbeddedFile        file;
//	DirList             list;
//	unsigned char       file_name[13];
//	unsigned int        size;
	char* toReturn;
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
	char* error03 = "ERR:file";
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

	while (ls_getNext(&list) == 0)
	{
		if ((list.currentEntry.FileName[8] == 'M') &&
			(list.currentEntry.FileName[9] == 'P') &&
			(list.currentEntry.FileName[10] == '3'))
		{
			DBG((TXT("Filename: %.11s (%li bytes)\n"), list.currentEntry.FileName, list.currentEntry.FileSize));

			format_file_name(file_name, list.currentEntry.FileName);

			if (file_fopen(&file, &efs.myFs, file_name, 'r') == 0)
			{
				file_fclose(&file);
				DBG((TXT("File successfully opened!.\n")));
				//TOGGLE_LIVE_LED1();
				return file_name;
			}
			else
			{
				DBG((TXT("Could not open file.\n")));
				return error03;
			}
		}
	}

	fs_umount(&(efs.myFs));

	return 0;
}
