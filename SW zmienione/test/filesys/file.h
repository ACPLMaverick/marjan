#ifndef __FILE_H_
#define __FILE_H_

/*****************************************************************************/
#include "config.h"
#include "error.h"
#include "time.h"
#include "fs.h"
#include "dir.h"
#include "libc.h"
#include "fat.h"
/*****************************************************************************/

#define MODE_READ 0x72
#define MODE_WRITE 0x77
#define MODE_APPEND 0x61

#define FILE_STATUS_OPEN 0
#define FILE_STATUS_WRITE 1

struct File{
	FileRecord DirEntry;		/*Copy of the FileRecord for this file */
	FileLocation Location;		/* Location in directory!! */
	FileSystem *fs;				/*Pointer to the filesystem this file is on */
	ClusterChain Cache;			/*Pointer to the cache object of the file  */
	unsigned char	FileStatus;			/*Contains bitfield regarding filestatus */
	unsigned long FilePtr;			/*Offsetpointer for fread/fwrite functions*/
	unsigned long FileSize;			/*Working copy of the filesize*/
};
typedef struct File File;


signed char file_fopen(File *file, FileSystem *fs,char *filename, char mode);
signed char file_fclose(File *file);
signed short file_setpos(File *file,unsigned long pos);
unsigned long file_fread(File *file,unsigned long offset, unsigned long size,unsigned char *buf);
unsigned long file_read (File *file,unsigned long size,unsigned char *buf);
unsigned long file_fwrite(File* file,unsigned long offset,unsigned long size,unsigned char* buf);
unsigned long file_write (File* file,unsigned long size,unsigned char* buf); 
char* file_normalToFatName(char* filename,char* fatfilename);
unsigned char file_validateChar(unsigned char c);
void file_initFile(File *file, FileSystem *fs, FileLocation *loc);
shortf file_allocClusterChain(File *file,unsigned long num_clusters);
void file_setAttr(File* file,unsigned char attribute,unsigned char val);
unsigned char file_getAttr(File* file,unsigned char attribute);
unsigned long file_requiredCluster(File *file,unsigned long offset, unsigned long size);

#endif
