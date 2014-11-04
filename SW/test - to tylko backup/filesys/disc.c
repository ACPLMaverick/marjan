/*****************************************************************************/
#include "disc.h"
/*****************************************************************************/

/* ****************************************************************************  
 * void disc_initDisc(Disc *disc,hcInterface* source)
 * Description: This initialises the disc by loading the MBR and setting the
 * pointer to the hardware object.
*/
void disc_initDisc(Disc *disc,IOManager* ioman)
{
	disc->ioman=ioman;
	disc_setError(disc,DISC_NOERROR);
	disc_loadMBR(disc);
}
/*****************************************************************************/ 

/* ****************************************************************************  
 * void disc_loadMBR(Disc *disc)
 * Description: This functions copies the partitiontable to the partitions field.
*/
void disc_loadMBR(Disc *disc)
{
	unsigned char x;
	unsigned char* buf;
	
	buf=ioman_getSector(disc->ioman,LBA_ADDR_MBR,IOM_MODE_READONLY|IOM_MODE_EXP_REQ);
	for(x=0;x<4;x++){
		ex_getPartitionField(buf,&(disc->partitions[x]),PARTITION_TABLE_OFFSET+(x*SIZE_PARTITION_FIELD));
	}
	ioman_releaseSector(disc->ioman,buf);
}
/*****************************************************************************/ 


