/*****************************************************************************/
#include "dir.h"
/*****************************************************************************/

/* ****************************************************************************  
 * void dir_getFileStructure(FileSystem *fs,FileRecord *filerec,FileLocation *loc)
 * Description: This function stores the filerecord located at loc in filerec.
 * It fetches the required sector for this.
 * Return value: void
*/
void dir_getFileStructure(FileSystem *fs,FileRecord *filerec,FileLocation *loc)
{
	unsigned char *buf;

	buf=part_getSect(fs->part,loc->Sector,IOM_MODE_READONLY);
	*filerec=*(((FileRecord*)buf)+loc->Offset);
	part_relSect(fs->part,buf);
}	

/*****************************************************************************/

/* ****************************************************************************  
 * void dir_createDirectoryEntry(FileSystem *fs,FileRecord *filerec,FileLocation *loc)
 * Description: This function writes the filerecord stored in filerec to disc at
 * location loc. 
 * Return value: void
*/
void dir_createDirectoryEntry(FileSystem *fs,FileRecord *filerec,FileLocation *loc)
{
	unsigned char *buf;
	
	buf = part_getSect(fs->part,loc->Sector,IOM_MODE_READWRITE);
	memCpy(filerec,buf+(loc->Offset*sizeof(*filerec)),sizeof(*filerec));
	part_relSect(fs->part,buf);
}
/*****************************************************************************/

/* ****************************************************************************  
 * void dir_createDefaultEntry(FileSystem *fs,FileRecord *filerec,char* fatfilename)
 * Description: This function fills in a filerecord with safe default values, and
 * a given fatfilename. If your system has a means of knowing time, here is an 
 * excellent place to apply it to the filerecord.  
 * Return value: void
*/
void dir_createDefaultEntry(FileSystem *fs,FileRecord *filerec,char* fatfilename)
{
	memCpy(fatfilename,filerec->FileName,11);
	filerec->Attribute=0x00;
	filerec->NTReserved=0x00;
	filerec->MilliSecTimeStamp=0x00;
	filerec->CreatedTime=time_getTime();
	filerec->CreatedDate=time_getDate(); 
	filerec->AccessDate=filerec->CreatedDate;
	filerec->FirstClusterHigh=0x0000;
	filerec->WriteTime=filerec->CreatedTime;
	filerec->WriteDate=filerec->CreatedDate;
	filerec->FirstClusterLow=0x0000;
	filerec->FileSize=0x00000000;
}
/*****************************************************************************/

/* ****************************************************************************  
 * void dir_setFirstCluster(File *file,unsigned long cluster_addr)
 * Description: This function requires modification to release it from
 * depending on the file object.
 * Return value:
*/
void dir_setFirstCluster(FileSystem *fs,FileLocation *loc,unsigned long cluster_addr)
{
	unsigned char *buf;
 	
 	buf = part_getSect(fs->part,loc->Sector,IOM_MODE_READWRITE);
	(((FileRecord*)buf)+loc->Offset)->FirstClusterHigh=cluster_addr>>16;
	(((FileRecord*)buf)+loc->Offset)->FirstClusterLow=cluster_addr&0xFFFF;	
	part_relSect(fs->part,buf);
}
/*****************************************************************************/

/* ****************************************************************************  
 * void dir_setFileSize(FileSystem *fs, FileLocation *loc,unsigned long numbytes)
 * Description: This function changes the filesize recorded at loc->Sector
 * to 'numbytes'.
 * Return value: void
*/
void dir_setFileSize(FileSystem *fs, FileLocation *loc,unsigned long numbytes)
{
	unsigned char *buf;
	
	buf = part_getSect(fs->part,loc->Sector,IOM_MODE_READWRITE);
	(((FileRecord*)buf)+loc->Offset)->FileSize=numbytes;
	part_relSect(fs->part,buf);
}
/*****************************************************************************/

/* ****************************************************************************  
 * signed char dir_updateDirectoryEntry(FileSystem *fs,FileRecord *entry,FileLocation *loc))
 * This function changes the entire entity stores at loc to the data recorded
 * in entry. This is for custom updates to the directoryentry.
 * Return value: 0 on success, -1 on failure
*/
signed char dir_updateDirectoryEntry(FileSystem *fs,FileRecord *entry,FileLocation *loc)
{
	unsigned char *buf;
	
	buf = part_getSect(fs->part,loc->Sector,IOM_MODE_READWRITE);
	memCpy(entry,buf+(loc->Offset*sizeof(*entry)),sizeof(*entry));
	part_relSect(fs->part,buf);
	return(0);
}

/* ****************************************************************************  
 * unsigned long dir_findFileinBuf(unsigned char *buf, char *fatname, FileLocation *loc)
 * This function searches for a given fatfilename in the buffer provided.
 * It will iterate through the 16 direntry's in the buffer and searches
 * for the fatfilename. If found, it will store the offset and attribute
 * entry of the directoryrecord in the loc structure.
 * If loc is 0, then it's members are not touched.
 * Return value: This function returns 0 when it cannot find the file,
 * if it can find the file it will return the first cluster number.
*/
unsigned long dir_findFileinBuf(unsigned char *buf, char *fatname, FileLocation *loc)
{
	FileRecord fileEntry;
	unsigned char c;
	
	for(c=0; c<16; c++)
	{
		fileEntry = *(((FileRecord*)buf) + c);
		/* Check if the entry is for short filenames */
		if( !( (fileEntry.Attribute & 0x0F) == 0x0F ) )
		{
			if( strMatch((char*)fileEntry.FileName,fatname,11) == 0 )
			{
				/* The entry has been found, return the location in the dir */
				if(loc)loc->Offset = c;
				if(loc)loc->attrib = fileEntry.Attribute;
				if((((unsigned long )fileEntry.FirstClusterHigh)<<16)+ fileEntry.FirstClusterLow==0){
					return(1); /* Lie about cluster, 0 means not found! */
				}else{
					return
							(
							(((unsigned long )fileEntry.FirstClusterHigh)<<16)
							+ fileEntry.FirstClusterLow
							);
				}
			}
		}
	}
	return(0);
}

/* ****************************************************************************  
 * unsigned long dir_findFreeEntryinBuf(unsigned char* buf, FileLocation *loc)
 * This function searches for a free entry in a given sector 'buf'.
 * It will put the offset into the loc->Offset field, given that loc is not 0.
 * Return value: 1 when it found a free spot, 0 if it hasn't.
*/
unsigned long dir_findFreeEntryinBuf(unsigned char* buf, FileLocation *loc)
{
	FileRecord fileEntry;
	unsigned char c;
	
	for(c=0;c<16;c++){
		fileEntry = *(((FileRecord*)buf) + c);
		if( !( (fileEntry.Attribute & 0x0F) == 0x0F ) ){
			if(fileEntry.FileName[0] == 0x00 ||
			   fileEntry.FileName[0] == 0xE5 ){
				if(loc)loc->Offset=c;
				return(1);
			}
		}
	}
	return(0);
}

/* ****************************************************************************  
 * unsigned long  dir_findinBuf(unsigned char *buf, char *fatname, FileLocation *loc)
 * Description: This function searches for a given fatfilename in a buffer.
 * Return value: Returns 0 on not found, and the firstcluster when the name is found.
*/
unsigned long  dir_findinBuf(unsigned char *buf, char *fatname, FileLocation *loc, unsigned char mode)
{
	switch(mode){
		case DIRFIND_FILE:
			return(dir_findFileinBuf(buf,fatname,loc));
			break;
		case DIRFIND_FREE:
			return(dir_findFreeEntryinBuf(buf,loc));
			break;
		default:
			return(0);
			break;
	}
	return(0);
}
/*****************************************************************************/

/* ****************************************************************************  
 * unsigned long dir_findinCluster(FileSystem *fs,unsigned long cluster,char *fatname, FileLocation *loc, unsigned char mode)
 * This function will search for an existing (fatname) or free directory entry
 * in a full cluster.
 * Return value: 0 on failure, firstcluster on finding file, and 1 on finding free spot.
*/
unsigned long dir_findinCluster(FileSystem *fs,unsigned long cluster,char *fatname, FileLocation *loc, unsigned char mode)
{
	unsigned char c,*buf=0;
	unsigned long fclus;
	
	for(c=0;c<fs->volumeId.SectorsPerCluster;c++){
		buf = part_getSect(fs->part,fs_clusterToSector(fs,cluster)+c,IOM_MODE_READONLY);
		if((fclus=dir_findinBuf(buf,fatname,loc,mode))){
			if(loc)loc->Sector=fs_clusterToSector(fs,cluster)+c;
			part_relSect(fs->part,buf);
			return(fclus);
		}
		part_relSect(fs->part,buf); /* Thanks Mike ;) */
	}
	return(0);
}

/* ****************************************************************************  
 * unsigned long dir_findinDir(FileSystem *fs, char* fatname,unsigned long firstcluster, FileLocation *loc, unsigned char mode)
 * This function will search for an existing (fatname) or free directory entry
 * in a directory, following the clusterchains.
 * Return value: 0 on failure, firstcluster on finding file, and 1 on finding free spot.
*/
unsigned long dir_findinDir(FileSystem *fs, char* fatname,unsigned long firstcluster, FileLocation *loc, unsigned char mode)
{
	unsigned long c=0,cluster;
	ClusterChain Cache;
	
	Cache.DiscCluster = Cache.FirstCluster = firstcluster;
	Cache.LogicCluster = Cache.LastCluster = Cache.Linear = 0;
	
	if(firstcluster <= 1){
		return(dir_findinRootArea(fs,fatname,loc,mode));	
	}
	
	while(!fat_LogicToDiscCluster(fs,&Cache,c++)){
		if((cluster=dir_findinCluster(fs,Cache.DiscCluster,fatname,loc,mode))){
			return(cluster);
		}
	}
	return(0);
}

/* ****************************************************************************  
 * unsigned long dir_findinDir(FileSystem *fs, char* fatname,unsigned long firstcluster, FileLocation *loc, unsigned char mode)
 * This function will search for an existing (fatname) or free directory entry
 * in the rootdirectory-area of a FAT12/FAT16 filesystem.
 * Return value: 0 on failure, firstcluster on finding file, and 1 on finding free spot.
*/
unsigned long dir_findinRootArea(FileSystem *fs,char* fatname, FileLocation *loc, unsigned char mode)
{
	unsigned long c,fclus;
	unsigned char *buf=0;
	
	if((fs->type != FAT12) && (fs->type != FAT16))return(0);
	
	for(c=fs->FirstSectorRootDir;c<(fs->FirstSectorRootDir+fs->volumeId.RootEntryCount/32);c++){
		buf = part_getSect(fs->part,c,IOM_MODE_READONLY);
		if((fclus=dir_findinBuf(buf,fatname,loc,mode))){
			if(loc)loc->Sector=c;
			part_relSect(fs->part,buf);
			return(fclus);
		}	
		part_relSect(fs->part,buf);	
	}
	part_relSect(fs->part,buf);
	return(0);
}

/* ****************************************************************************  
 * signed char dir_getFatFileName(char* filename, char* fatfilename)
 * This function will take a full directory path, and strip off all leading
 * dirs and characters, leaving you with the MS-DOS notation of the actual filename.
 * Return value: 1 on success, 0 on not being able to produca a filename
*/
signed char dir_getFatFileName(char* filename, char* fatfilename)
{
	char ffnamec[11],*next,nn=0;
	
	memClr(ffnamec,11); memClr(fatfilename,11);
	next = filename;
	
	if(*filename=='/')next++;
	
	while((next=file_normalToFatName(next,ffnamec))){
		memCpy(ffnamec,fatfilename,11);	
		nn++;
	}
	if(nn)return(1);
	return(0);
}

/* ****************************************************************************  
 * signed char dir_addCluster(FileSystem *fs,unsigned long firstCluster)
 * This function extends a directory by 1 cluster + optional the number of
 * clusters you want pre-allocated. It will also delete the contents of that
 * cluster. (or clusters)
 * Return value: 0 on success, -1 on fail
*/
signed char dir_addCluster(FileSystem *fs,unsigned long firstCluster)
{
	unsigned long lastc,logicalc;
	ClusterChain cache;
		
	fs_initClusterChain(fs,&cache,firstCluster);
	if(fat_allocClusterChain(fs,&cache,1)){
		return(-1);
	}
	lastc = fs_getLastCluster(fs,&cache);
	if(CLUSTER_PREALLOC_DIRECTORY){
		if(fat_allocClusterChain(fs,&cache,CLUSTER_PREALLOC_DIRECTORY)){
			return(-1);
		}
		logicalc = fat_DiscToLogicCluster(fs,firstCluster,lastc);
		while(!fat_LogicToDiscCluster(fs,&cache,++logicalc)){
			fs_clearCluster(fs,cache.DiscCluster);
		}
	}else{
			fs_clearCluster(fs,lastc);
	}
	return(0);
}


