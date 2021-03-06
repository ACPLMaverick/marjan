/*****************************************************************************/
#include "ls.h"
/*****************************************************************************/

signed char ls_openDir(DirList *dlist,FileSystem *fs,char* dirname)
{
	FileLocation loc;
	unsigned long fc;
	
	dlist->fs=fs;
	
	if(fs_findFile(dlist->fs,dirname,&loc,&fc)!=2)
	{
		return(-1);
	}
	
	fs_initClusterChain(dlist->fs,&(dlist->Cache),fc);
	memClr(&(dlist->currentEntry),sizeof(dlist->currentEntry));
	dlist->rEntry=0;
	dlist->cEntry=0xFFFF;
	
	return(0);
}
/*****************************************************************************/

signed char ls_getDirEntry(DirList *dlist)
{
	if(dlist->Cache.FirstCluster == 1){
		return(ls_getRootAreaEntry(dlist));
	}else if(dlist->Cache.FirstCluster){
		return(ls_getRealDirEntry(dlist));
	}
	return(-1);
}
/*****************************************************************************/

signed char ls_getNext(DirList *dlist)
{
	do{
		if(ls_getDirEntry(dlist))return(-1);
		dlist->rEntry++;
	}while(!ls_isValidFileEntry(&(dlist->currentEntry)));
	dlist->cEntry++;
	return(0);
}
/*****************************************************************************/

signed char ls_getRealDirEntry(DirList *dlist)
{
	unsigned char* buf;
	
	if(dlist->Cache.FirstCluster<=1)return(-1);
	
	if(fat_LogicToDiscCluster(dlist->fs,
						   &(dlist->Cache),
						   (dlist->rEntry)/(16 * dlist->fs->volumeId.SectorsPerCluster))){
		return(-1);
	}
	
	buf = part_getSect(dlist->fs->part,
					   fs_clusterToSector(dlist->fs,dlist->Cache.DiscCluster) + (dlist->rEntry/16)%dlist->fs->volumeId.SectorsPerCluster,
				       IOM_MODE_READONLY);
	
	/*memCpy(buf+(dlist->rEntry%16)*32,&(dlist->currentEntry),32);*/
	ls_fileEntryToDirListEntry(dlist,buf,32*(dlist->rEntry%16));
	
	part_relSect(dlist->fs->part,buf);
	
	return(0);
}
/*****************************************************************************/

signed char ls_getRootAreaEntry(DirList *dlist)
{
	unsigned char *buf=0;
	
	if((dlist->fs->type != FAT12) && (dlist->fs->type != FAT16))return(-1);
	if(dlist->rEntry>=dlist->fs->volumeId.RootEntryCount)return(-1);
	
	buf = part_getSect(dlist->fs->part,
					   dlist->fs->FirstSectorRootDir+dlist->rEntry/16,
					   IOM_MODE_READONLY);
	/*memCpy(buf+32*(dlist->rEntry%16),&(dlist->currentEntry),32);*/
	ls_fileEntryToDirListEntry(dlist,buf,32*(dlist->rEntry%16));
	part_relSect(dlist->fs->part,buf);
	return(0);
}
/*****************************************************************************/

signed char ls_isValidFileEntry(ListDirEntry *entry)
{
	if(entry->FileName[0] == 0 || entry->FileName[0] == 0xE5 || entry->FileName[0] == '.')return(0);
	if((entry->Attribute&0x0F)==0x0F)return(0);
 	return(1);
}
/*****************************************************************************/

void ls_fileEntryToDirListEntry(DirList *dlist, unsigned char* buf, unsigned short offset)
{
	if(offset>480 || offset%32)return;
	
	buf+=offset;
	memCpy(buf+OFFSET_DE_FILENAME,dlist->currentEntry.FileName,LIST_MAXLENFILENAME);
	dlist->currentEntry.Attribute = *(buf+OFFSET_DE_ATTRIBUTE);
	dlist->currentEntry.FileSize = ex_getb32(buf,OFFSET_DE_FILESIZE);
}
/*****************************************************************************/

