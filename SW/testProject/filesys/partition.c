/*****************************************************************************/
#include "partition.h"
/*****************************************************************************/

/* ****************************************************************************  
 * void part_initPartition(Partition *part,Disc* refDisc)
 * Description: This function searches the 4 partitions for a FAT class partition
 * and marks the first one found as the active to be used partition.
*/
void part_initPartition(Partition *part,Disc* refDisc)
{
	short c;
	
	part->disc=refDisc;
	part->activePartition=-1; /* No partition selected */
	part_setError(part,PART_NOERROR);
	for(c=3;c>=0;c--){
		if(part_isFatPart(part->disc->partitions[c].type))
			part->activePartition=c;
	} 
}
/*****************************************************************************/ 


/* ****************************************************************************  
 * short part_isFatPart(unsigned char type)
 * Description: This functions checks if a partitiontype (char) is of the FAT
 * type in the broadest sense. I
 * Return value: If it is FAT, returns 1, otherwise 0.
*/
short part_isFatPart(unsigned char type)
{
	if(type == PT_FAT12  ||
	   type == PT_FAT16A ||
	   type == PT_FAT16  ||
	   type == PT_FAT32  ||
	   type == PT_FAT32A ||
	   type == PT_FAT16B   )
	{
		return(1);
	}
	return(0);
}
/*****************************************************************************/ 

signed char part_readBuf(Partition *part, unsigned long address, unsigned char* buf)
{
	return(if_readBuf(part->disc->ioman->iface,part_getRealLBA(part,address), buf));
}

/* ****************************************************************************  
 * short part_writeBuf(Partition *part,unsigned long address,unsigned char* buf)
 * Description: This function writes 512 bytes, from buf. It's offset is address
 * sectors from the beginning of the partition.
 * Return value: It returns whatever the hardware function returns. (-1=error)
*/
short part_writeBuf(Partition *part,unsigned long address,unsigned char* buf)
{
	/*DBG((TXT("part_writeBuf :: %li\n"),address));*/
	return(if_writeBuf(part->disc->ioman->iface,part_getRealLBA(part,address),buf));
}
/*****************************************************************************/ 


/* ****************************************************************************  
 * unsigned long part_getRealLBA(Partition *part,unsigned long address)
 * Description: This function calculates what the partition offset for
 * a partition is + the address.
 * Return value: Sector address.
*/
unsigned long part_getRealLBA(Partition *part,unsigned long address)
{
	return(part->disc->partitions[part->activePartition].LBA_begin+address);
}
/*****************************************************************************/ 

/* ****************************************************************************  
 * unsigned char* part_getSect(Partition *part, unsigned long address, unsigned char mode)
 * Description: This function calls ioman_getSector, but recalculates the sector
 * address to be partition relative.
 * Return value: Whatever getSector returns. (pointer or 0)
*/
unsigned char* part_getSect(Partition *part, unsigned long address, unsigned char mode)
{
	return(ioman_getSector(part->disc->ioman,part_getRealLBA(part,address),mode));
}

/* ****************************************************************************  
 * signed char part_relSect(Partition *part, unsigned char* buf)
 * Description: This function calls ioman_releaseSector.
 * Return value: Whatever releaseSector returns.
*/
signed char part_relSect(Partition *part, unsigned char* buf)
{
	return(ioman_releaseSector(part->disc->ioman,buf));
}

signed char part_flushPart(Partition *part,unsigned long addr_l, unsigned long addr_h)
{
	return( 
		ioman_flushRange(part->disc->ioman,part_getRealLBA(part,addr_l),part_getRealLBA(part,addr_h)) 
	);	
}

signed char part_directSectorRead(Partition *part,unsigned long address, unsigned char* buf)
{
	return(
		ioman_directSectorRead(part->disc->ioman,part_getRealLBA(part,address),buf)
	);
}

signed char part_directSectorWrite(Partition *part,unsigned long address, unsigned char* buf)
{
	return(
		ioman_directSectorWrite(part->disc->ioman,part_getRealLBA(part,address),buf)
	);
}


