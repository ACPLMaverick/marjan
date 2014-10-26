
/*****************************************************************************/
#include "efs.h"
/*****************************************************************************/

/* ****************************************************************************  
 * esint8 efs_init(EmbeddedFileSystem * efs,char* opts)
 * Description: This function initialises all subelements of a filesystem.
 * It sets the pointerchain and verifies each step.
 * Return value: 0 on success and -1 on failure.
*/
signed int efs_init(EmbeddedFileSystem * efs,char* opts)
{
	if(if_initInterface(&efs->myCard, opts)==0)
	{
		ioman_init(&efs->myIOman,&efs->myCard,0);
		disc_initDisc(&efs->myDisc, &efs->myIOman);
		part_initPartition(&efs->myPart, &efs->myDisc);
		if(efs->myPart.activePartition==-1){
			efs->myDisc.partitions[0].type=0x0B;
			efs->myDisc.partitions[0].LBA_begin=0;
			efs->myDisc.partitions[0].numSectors=efs->myCard.sectorCount;	
			/*efs->myPart.activePartition = 0;*/
			/*efs->myPart.disc = &(efs->myDisc);*/
			part_initPartition(&efs->myPart, &efs->myDisc);
		}
		/*part_initPartition(&efs->myPart, &efs->myDisc);*/
		if(fs_initFs(&efs->myFs, &efs->myPart))
			return(-2);
		return(0);
	}
	return(-1);
}
/*****************************************************************************/


