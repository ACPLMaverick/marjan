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


void testMMC(void)
{
	if (efs_init(&efs, "\\") != 0)
	{
		//DBG((TXT("Could not open filesystem.\n")));
		return(-1);
	}

	if (ls_openDir(&list, &(efs.myFs), "/") != 0)
	{
		//DBG((TXT("Could not open the selected directory.\n")));			//Trzeba to zmieniæ na coœ innego bo to jest w debug.h a tego nie ma
		return(-2);
	}

	while (ls_getNext(&list) == 0)
	{
		if ((list.currentEntry.FileName[8] == 'M') &&
			(list.currentEntry.FileName[9] == 'P') &&
			(list.currentEntry.FileName[10] == '3'))
		{
			//DBG((TXT("Filename: %.11s (%li bytes)\n"), list.currentEntry.FileName,
				list.currentEntry.FileSize));

			format_file_name(file_name, list.currentEntry.FileName);

			if (file_fopen(&file, &efs.myFs, file_name, 'r') == 0)
			{
				file_fclose(&file);
				TOGGLE_LIVE_LED1();
			}
			//else
				//DBG((TXT("Could not open file.\n")));
		}
	}

	fs_umount(&(efs.myFs));

	return 0;
}
