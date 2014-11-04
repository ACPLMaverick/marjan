#ifndef __PARTITION_H__
#define __PARTITION_H__

/*****************************************************************************/
#include "config.h"
#include "error.h"
#include "../mmc/interface.h"
#include "disc.h"
/*****************************************************************************/

#define PT_FAT12  0x01
#define PT_FAT16A 0x04
#define PT_FAT16  0x06
#define PT_FAT32  0x0B
#define PT_FAT32A 0x5C
#define PT_FAT16B 0x5E

struct Partition{
	Disc *disc;						/*Pointer to disc containing this partition.*/
	signed char activePartition;	/*Array subscript for disc->partitions[activePartition]*/
};
typedef struct Partition Partition;

void part_initPartition(Partition *part,Disc* refDisc);
short part_isFatPart(unsigned char type);
signed char part_readBuf(Partition *part, unsigned long address, unsigned char* buf);
signed char part_readPartBuf(Partition *part, unsigned long address, unsigned char* buf, unsigned long offset, unsigned long len);
short part_writeBuf(Partition *part,unsigned long address,unsigned char* buf);
unsigned char* part_getSect(Partition *part, unsigned long address,unsigned char mode);
signed char part_relSect(Partition *part, unsigned char* buf);
signed char part_flushPart(Partition *part,unsigned long addr_l, unsigned long addr_h);
signed char part_directSectorRead(Partition *part, unsigned long address, unsigned char* buf);
signed char part_directSectorWrite(Partition *part, unsigned long address, unsigned char* buf);
unsigned long part_getRealLBA(Partition *part,unsigned long address);

#include "extract.h"

#endif
