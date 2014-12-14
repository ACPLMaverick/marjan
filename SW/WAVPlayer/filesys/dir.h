#ifndef __DDIR_H__
#define __DDIR_H__

/*****************************************************************************/
#include "config.h"
#include "error.h"
#include "fat.h"
#include "libc.h"
#include "ioman.h"
#include "time.h"
#include "fs.h"
/*****************************************************************************/

#define ATTR_READ_ONLY  0x01
#define ATTR_HIDDEN     0x02
#define ATTR_SYSTEM     0x04
#define ATTR_VOLUME_ID  0x08
#define ATTR_DIRECTORY  0x10
#define ATTR_ARCHIVE    0x20

#define OFFSET_DE_FILENAME            0
#define OFFSET_DE_ATTRIBUTE          11
#define OFFSET_DE_NTRESERVED         12
#define OFFSET_DE_CRTIMETNT          13
#define OFFSET_DE_CREATETIME         14
#define OFFSET_DE_CREATEDATE         16
#define OFFSET_DE_LASTACCESSDATE     18
#define OFFSET_DE_FIRSTCLUSTERHIGH   20
#define OFFSET_DE_WRITETIME          22
#define OFFSET_DE_WRITEDATE          24
#define OFFSET_DE_FIRSTCLUSTERLOW    26
#define OFFSET_DE_FILESIZE           28

#define DIRFIND_FILE 0
#define DIRFIND_FREE 1

void dir_getFileStructure(FileSystem *fs,FileRecord *filerec,FileLocation *loc);
void dir_createDirectoryEntry(FileSystem *fs,FileRecord *filerec,FileLocation *loc);
void dir_createDefaultEntry(FileSystem *fs,FileRecord *filerec,char* fatfilename);
void dir_setFirstCluster(FileSystem *fs,FileLocation *loc,unsigned long cluster_addr);
void dir_setFileSize(FileSystem *fs,FileLocation *loc,unsigned long numbytes);
unsigned long dir_findinRoot(FileSystem *fs,char * fatname, FileLocation *loc);
unsigned long dir_findinDir(FileSystem *fs, char * fatname, unsigned long startCluster, FileLocation *loc, unsigned char mode);
unsigned long dir_findinBuf(unsigned char *buf,char *fatname, FileLocation *loc, unsigned char mode);
unsigned long dir_findinCluster(FileSystem *fs,unsigned long cluster,char *fatname, FileLocation *loc, unsigned char mode);
unsigned long dir_findinRootArea(FileSystem *fs,char* fatname, FileLocation *loc, unsigned char mode);
signed char dir_getFatFileName(char* filename, char* fatfilename);
signed char dir_updateDirectoryEntry(FileSystem *fs,FileRecord *entry,FileLocation *loc);
signed char dir_addCluster(FileSystem *fs,unsigned long firstCluster);
#endif
