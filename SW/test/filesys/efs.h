#ifndef __EFS_H__
#define __EFS_H__

/*****************************************************************************/
#include "config.h"
#include "../mmc/interface.h"
#include "disc.h"
#include "partition.h"
#include "fs.h"
#include "file.h"
#include "time.h"
#include "./ui.h"
/*****************************************************************************/

typedef File EmbeddedFile;

struct EmbeddedFileSystem{
	hwInterface myCard;
	IOManager myIOman;
	Disc myDisc;
	Partition myPart;
	FileSystem myFs;
};
typedef struct  EmbeddedFileSystem EmbeddedFileSystem;

signed int efs_init(EmbeddedFileSystem * efs, char* opts);


#endif

