#ifndef __LS_H__
#define __LS_H__

/*****************************************************************************/
#include "config.h"
#include "fs.h"
#include "dir.h"
#include "fat.h"
/*****************************************************************************/

struct ListDirEntry{
	unsigned char FileName[LIST_MAXLENFILENAME];
	unsigned long FileSize;
	unsigned char Attribute;
};
typedef struct ListDirEntry ListDirEntry;

struct DirList{
	FileSystem *fs;
	unsigned short cEntry,rEntry;
	/*FileRecord currentEntry;*/
	ListDirEntry currentEntry;
	ClusterChain Cache;
};
typedef struct DirList DirList;

signed char ls_openDir(DirList *dlist,FileSystem *fs,char* dirname);
signed char ls_getNext(DirList *dlist);

signed char ls_getDirEntry(DirList *dlist);
signed char ls_getRealDirEntry(DirList *dlist);
signed char ls_getRootAreaEntry(DirList *dlist);
signed char ls_isValidFileEntry(ListDirEntry *entry);
void   ls_fileEntryToDirListEntry(DirList *dlist, unsigned char* buf, unsigned short offset);

#endif
