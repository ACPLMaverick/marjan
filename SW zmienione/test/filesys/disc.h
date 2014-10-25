#ifndef __DISC_H_
#define __DISC_H_

/*****************************************************************************/
#include "config.h"
#include "error.h"
#include "../mmc/interface.h"
#include "ioman.h"
/*****************************************************************************/

#define LBA_ADDR_MBR 0
#define PARTITION_TABLE_OFFSET 0x1BE

struct PartitionField{
	unsigned char bootFlag;
	unsigned char CHS_begin[3];
	unsigned char type;			/*Type of partition*/
	unsigned char CHS_end[3];
	unsigned long LBA_begin;	/*LBA address of first sector*/
	unsigned long numSectors;	/*Numer of 512byte sectors*/
};
typedef struct  PartitionField  PartitionField;

#define SIZE_PARTITION_FIELD 16

struct Disc{
	IOManager *ioman;
	DISC_ERR_EUINT8
		PartitionField partitions[4];	/*Array of PartitionFields, containing the partition info*/
};
typedef struct Disc Disc;

void disc_initDisc(Disc *disc,IOManager *ioman);
void disc_loadMBR(Disc *disc);

#endif
