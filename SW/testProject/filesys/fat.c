/*****************************************************************************/
#include "fs.h"
/*****************************************************************************/

/* ****************************************************************************  
 * unsigned long fat_getSectorAddressFatEntry(FileSystem *fs,unsigned long cluster_addr)
 * Description: Returns the sectornumber that holds the fat entry for cluster cluster_addr.
 * This works for all FAT types.
 * Return value: Sectornumber, or 0. Warning, no boundary check.
*/
unsigned long fat_getSectorAddressFatEntry(FileSystem *fs,unsigned long cluster_addr)
{ 
	unsigned long base = fs->volumeId.ReservedSectorCount,res;
	
	switch(fs->type){
		case FAT12:
			res=(cluster_addr*3/1024);
			if(res>=fs->FatSectorCount){
				return(0);
			}else{
				return(base+res);
			}
			break;
		case FAT16:
			res=cluster_addr/256;
			if(res>=fs->FatSectorCount){
				return(0);
			}else{
				return(base+res);
			}
			break;
		case FAT32:
			res=cluster_addr/128;
			if(res>=fs->FatSectorCount){
				return(0);
			}else{
				return(base+res);
			}
			break; 
	}
	return(0);
}
/*****************************************************************************/ 


/* ****************************************************************************  
 * unsigned long fat_getNextClusterAddress(FileSystem *fs,unsigned long cluster_addr
 * Description: This function loads the sector of the fat which contains the entry
 * for cluster_addr. It then fetches and (if required) calculates it's value.
 * This value is the EoC marker -or- the number of the next cluster in the chain.
 * Return value: Clusternumber or EoC
*/
unsigned long fat_getNextClusterAddress(FileSystem *fs,unsigned long cluster_addr,unsigned short *linear)
{
	unsigned char *buf; 
	unsigned char hb,lb;
	unsigned short offset;
	unsigned long sector;
	unsigned long nextcluster=0;
	
	sector=fat_getSectorAddressFatEntry(fs,cluster_addr);
	if( (fs->FatSectorCount <= (sector-fs->volumeId.ReservedSectorCount)) || sector==0 )
	{
		return(0);
	}
	
	buf=part_getSect(fs->part,sector,IOM_MODE_READONLY);
		
	switch(fs->type)
	{
		case FAT12:
			offset = ((cluster_addr%1024)*3/2)%512;
			hb = buf[offset];
			if(offset == 511){
				part_relSect(fs->part,buf);
				buf=part_getSect(fs->part,sector+1,IOM_MODE_READONLY);
				lb = buf[0];
			}else{
				lb = buf[offset + 1];
			}
			if(cluster_addr%2==0){
				nextcluster = ( ((lb&0x0F)<<8) + (hb) );
			}else{
				nextcluster = ( (lb<<4) + (hb>>4) );
			}
			break;
		case FAT16:
			offset=cluster_addr%256;
			nextcluster = *((unsigned short *)buf + offset);
			break;
		case FAT32:
			offset=cluster_addr%128;
			nextcluster = *((unsigned long *)buf + offset);
			break;
	}
	
	part_relSect(fs->part,buf);
	
	return(nextcluster);
}
/*****************************************************************************/ 


/* ****************************************************************************  
 * void fat_setNextClusterAddress(FileSystem *fs,unsigned long cluster_addr,unsigned long next_cluster_addr)
 * Description: This function makes an entry in the fattable for cluster_addr. The value it puts there
 * is next_cluster_addr. 
*/
void fat_setNextClusterAddress(FileSystem *fs,unsigned long cluster_addr,unsigned long next_cluster_addr)
{
	unsigned char *buf,*buf2; 
	unsigned short offset;
	unsigned long sector;
	
	sector=fat_getSectorAddressFatEntry(fs,cluster_addr);
	
	if(( fs->FatSectorCount <= (sector - fs->volumeId.ReservedSectorCount )||(sector==0))){
	    DBG((TXT("HARDERROR:::fat_getNextClusterAddress READ PAST FAT BOUNDARY\n")));
	    return;
	}
	
	buf=part_getSect(fs->part,sector,IOM_MODE_READWRITE);
		
	switch(fs->type){
		case FAT12:
			offset = ((cluster_addr%1024)*3/2)%512;
			if(offset == 511){
				if(cluster_addr%2==0){
					buf[offset]=next_cluster_addr&0xFF;
				}else{
					buf[offset]=(buf[offset]&0xF)+((next_cluster_addr<<4)&0xF0);
				}
				buf2=part_getSect(fs->part,fat_getSectorAddressFatEntry(fs,cluster_addr)+1,IOM_MODE_READWRITE);
				if(cluster_addr%2==0){
					buf2[0]=(buf2[0]&0xF0)+((next_cluster_addr>>8)&0xF);
				}else{
					buf2[0]=(next_cluster_addr>>4)&0xFF;
				}
				part_relSect(fs->part,buf2);
			}else{
				if(cluster_addr%2==0){
					buf[offset]=next_cluster_addr&0xFF;
					buf[offset+1]=(buf[offset+1]&0xF0)+((next_cluster_addr>>8)&0xF);
				}else{
					buf[offset]=(buf[offset]&0xF)+((next_cluster_addr<<4)&0xF0);
					buf[offset+1]=(next_cluster_addr>>4)&0xFF;
				}
			}
			part_relSect(fs->part,buf);
			break;
		case FAT16:
			offset=cluster_addr%256;
			*((unsigned short*)buf+offset)=next_cluster_addr;
			part_relSect(fs->part,buf);
			break;
		case FAT32:
			offset=cluster_addr%128;
			*((unsigned long*)buf+offset)=next_cluster_addr;
			part_relSect(fs->part,buf);
			break;
	}
	
}
/*****************************************************************************/ 


/* ****************************************************************************  
 * short fat_isEocMarker(FileSystem *fs,unsigned long fat_entry)
 * Description: Checks if a certain value is the EoC marker for the filesystem
 * noted in fs->type.
 * Return value: Returns 0 when it is the EoC marker, and 1 otherwise.
*/
short fat_isEocMarker(FileSystem *fs,unsigned long fat_entry)
{
	switch(fs->type){
		case FAT12:
			if(fat_entry<0xFF8){
				return(0);
			}
			break;
		case FAT16:
			if(fat_entry<0xFFF8){
				return(0);
			}
			break;
		case FAT32:
			if((fat_entry&0x0FFFFFFF)<0xFFFFFF8){
				return(0);
			}
			break;
	}
	return(1);
}
/*****************************************************************************/ 


/* ****************************************************************************  
 * unsigned long fat_giveEocMarker(FileSystem *fs)
 * Description: Returns an EoC markernumber valid for the filesystem noted in
 * fs->type.
 * Note, for FAT32, the upper 4 bits are set to zero, although they should be un
 * touched according to MicroSoft specifications. I didn't care.
 * Return value: The EoC marker cast to an ulong.
*/
unsigned long fat_giveEocMarker(FileSystem *fs)
{
	switch(fs->type)
	{
		case FAT12:
			return(0xFFF);
			break;
		case FAT16:
			return(0xFFFF);
			break;
		case FAT32:
			return(0x0FFFFFFF);
			break;
	}
	return(0);
}
/*****************************************************************************/ 

/* ****************************************************************************  
 * unsigned long fat_getNextClusterAddressWBuf(FileSystem *fs,unsigned long cluster_addr, unsigned char* buf)
 * Description: This function retrieves the contents of a FAT field. It does not fetch
 * it's own buffer, it is given as a parameter. (ioman makes this function rather obsolete)
 * Only in the case of a FAT12 crosssector data entry a sector is retrieved here.
 * Return value: The value of the clusterfield is returned.
*/
unsigned long fat_getNextClusterAddressWBuf(FileSystem *fs,unsigned long cluster_addr, unsigned char* buf)
{
	unsigned char  *buf2; /* For FAT12 fallover only */
	unsigned char hb,lb;
	unsigned short offset;
	unsigned long nextcluster=0;
	
	switch(fs->type)
	{
		case FAT12:
			offset = ((cluster_addr%1024)*3/2)%512;
			hb = buf[offset];
			if(offset == 511){
				buf2=part_getSect(fs->part,fat_getSectorAddressFatEntry(fs,cluster_addr)+1,IOM_MODE_READONLY);
				lb = buf2[0];
				part_relSect(fs->part,buf2);
			}else{
				lb = buf[offset + 1];
			}
			if(cluster_addr%2==0){
				nextcluster = ( ((lb&0x0F)<<8) + (hb) );
			}else{
				nextcluster = ( (lb<<4) + (hb>>4) );
			}
			break;
		case FAT16:
			offset=cluster_addr%256;
			nextcluster = *((unsigned short*)buf + offset);
			break;
		case FAT32:
			offset=cluster_addr%128;
			nextcluster = *((unsigned long*)buf + offset);
			break;
	}
	return(nextcluster);
}
/*****************************************************************************/ 

/* ****************************************************************************  
 * void fat_setNextClusterAddressWBuf(FileSystem *fs,unsigned long cluster_addr,unsigned long next_cluster_addr,unsigned char* buf)
 * Description: This function fills in a fat entry. The entry is cluster_addr and the
 * data entered is next_cluster_addr. This function is also given a *buf, so it does
 * not write the data itself, except in the case of FAT 12 cross sector data, where
 * the second sector is handled by this function.
 * Return value:
*/
void fat_setNextClusterAddressWBuf(FileSystem *fs,unsigned long cluster_addr,unsigned long next_cluster_addr,unsigned char* buf)
{
	unsigned short offset;
	unsigned char *buf2;
		
	switch(fs->type)
	{
		case FAT12:
			offset = ((cluster_addr%1024)*3/2)%512;
			if(offset == 511){
				if(cluster_addr%2==0){
					buf[offset]=next_cluster_addr&0xFF;
				}else{
					buf[offset]=(buf[offset]&0xF)+((next_cluster_addr<<4)&0xF0);
				}
				buf2=part_getSect(fs->part,fat_getSectorAddressFatEntry(fs,cluster_addr)+1,IOM_MODE_READWRITE);
				if(cluster_addr%2==0){
					buf2[0]=(buf2[0]&0xF0)+((next_cluster_addr>>8)&0xF);
				}else{
					buf2[0]=(next_cluster_addr>>4)&0xFF;
				}
				part_relSect(fs->part,buf2);
			}else{
				if(cluster_addr%2==0){
					buf[offset]=next_cluster_addr&0xFF;
					buf[offset+1]=(buf[offset+1]&0xF0)+((next_cluster_addr>>8)&0xF);
				}else{
					buf[offset]=(buf[offset]&0xF)+((next_cluster_addr<<4)&0xF0);
					buf[offset+1]=(next_cluster_addr>>4)&0xFF;
				}
			}
			break;
		case FAT16:
			offset=cluster_addr%256;
			*((unsigned short*)buf+offset)=next_cluster_addr;
			break;
		case FAT32:
			offset=cluster_addr%128;
			*((unsigned long*)buf+offset)=next_cluster_addr;
			break;
	}
}
/*****************************************************************************/

/* ****************************************************************************  
 * signed short fat_getNextClusterChain(FileSystem *fs, ClusterChain *Cache)
 * Description: This function is to advance the clusterchain of a Cache.
 * First, the function verifies if the Cache is valid. It could correct it if it 
 * is not, but this is not done at the time. If the cachen is valid, the next step is
 * to see what the next cluster is, if this is the End of Clustermark, the cache is
 * updated to know the lastcluster but will remain untouched otherwise. -1 is returned.
 * If there are more clusters the function scans the rest of the chain until the next
 * cluster is no longer lineair, or until it has run out of fat data (only 1 sector) is
 * examined, namely the one fetched to check for EoC.
 * With lineair is meant that logical cluster n+1 should be 1 more than logical cluster n
 * at the disc level.
 * Return value: 0 on success, or -1 when EoC.
*/
signed short fat_getNextClusterChain(FileSystem *fs, ClusterChain *Cache)
{
	unsigned long sect,lr,nlr,dc;
	signed short lin=0;
	unsigned char *buf;

	if(Cache->DiscCluster==0)
	{
		return(-1);
	}

	sect=fat_getSectorAddressFatEntry(fs,Cache->DiscCluster);
	buf=part_getSect(fs->part,sect,IOM_MODE_READONLY);
	dc=fat_getNextClusterAddressWBuf(fs,Cache->DiscCluster,buf);
	if(fat_isEocMarker(fs,dc))
	{
		Cache->LastCluster=Cache->DiscCluster;
		part_relSect(fs->part,buf);
		return(-1);
	}
	
	Cache->DiscCluster=dc;
	Cache->LogicCluster++;
		
	lr=Cache->DiscCluster-1;
	nlr=lr+1;
	
	while(nlr-1==lr && fat_getSectorAddressFatEntry(fs,nlr)==sect)
	{
		lr=nlr;
		nlr=fat_getNextClusterAddressWBuf(fs,lr,buf);
		lin++;	
	}
	
	Cache->Linear=lin-1<0?0:lin-1;
	
	part_relSect(fs->part,buf);
	return(0);
}
/*****************************************************************************/


/* ****************************************************************************  
 * signed short fat_LogicToDiscCluster(FileSystem *fs, ClusterChain *Cache,unsigned long logiccluster)
 * Description: This function is used to follow clusterchains. When called it will convert
 * a logical cluster, to a disc cluster, using a Cache object. All it does is call
 * getNextClusterChain in the proper manner, and rewind clusterchains if required.
 * It is NOT recommended to go backwards in clusterchains, since this will require
 * scanning the entire chain every time.
 * Return value: 0 on success and -1 on failure (meaning out of bounds).
*/
signed short fat_LogicToDiscCluster(FileSystem *fs, ClusterChain *Cache,unsigned long logiccluster)
{
	if(logiccluster<Cache->LogicCluster || Cache->DiscCluster==0){
		Cache->LogicCluster=0;
		Cache->DiscCluster=Cache->FirstCluster;
		Cache->Linear=0;
	}
	
	if(Cache->LogicCluster==logiccluster){
		return(0);
	}
	
	while(Cache->LogicCluster!=logiccluster)
	{
		if(Cache->Linear!=0)
		{
			Cache->Linear--;
			Cache->LogicCluster++;
			Cache->DiscCluster++;
		}
		else
		{
			if((fat_getNextClusterChain(fs,Cache))!=0){
				return(-1);
			}
		}
	}
	return(0);
}
/*****************************************************************************/

/* ****************************************************************************  
 * short fat_allocClusterChain(FileSystem *fs,ClusterChain *Cache,unsigned long num_clusters)
 * Description: This function extends a clusterchain by num_clusters. It returns the
 * number of clusters it *failed* to allocate. 
 * Return value: 0 on success, all other values are the number of clusters it could
 * not allocate.
*/
short fat_allocClusterChain(FileSystem *fs,ClusterChain *Cache,unsigned long num_clusters)
{
	unsigned long cc,ncl=num_clusters,lc;
	unsigned char *bufa=0,*bufb=0;
	unsigned char  overflow=0;

	if(Cache->FirstCluster<=1)return(num_clusters);
	
	lc=fs_getLastCluster(fs,Cache);
	cc=lc;
	
	while(ncl > 0){
		cc++;
		if(cc>=fs->DataClusterCount+1){
			if(overflow){
				bufa=part_getSect(fs->part,fat_getSectorAddressFatEntry(fs,lc),IOM_MODE_READWRITE);
				fat_setNextClusterAddressWBuf(fs,lc,fat_giveEocMarker(fs),bufa);
				Cache->LastCluster=lc;
				part_relSect(fs->part,bufa);
				fs->FreeClusterCount-=num_clusters-ncl;
				return(num_clusters-ncl);
			}
			cc=2;
			overflow++;
		}
		bufa=part_getSect(fs->part,fat_getSectorAddressFatEntry(fs,cc),IOM_MODE_READONLY);
		if(fat_getNextClusterAddressWBuf(fs,cc,bufa)==0){
			bufb=part_getSect(fs->part,fat_getSectorAddressFatEntry(fs,lc),IOM_MODE_READWRITE);
			fat_setNextClusterAddressWBuf(fs,lc,cc,bufb);
			part_relSect(fs->part,bufb);
			ncl--;
			lc=cc;
		}
		part_relSect(fs->part,bufa);
		if(ncl==0){
			bufa=part_getSect(fs->part,fat_getSectorAddressFatEntry(fs,lc),IOM_MODE_READWRITE);
			fat_setNextClusterAddressWBuf(fs,lc,fat_giveEocMarker(fs),bufa);
			Cache->LastCluster=lc;
			part_relSect(fs->part,bufa);
		}
	}
	if(Cache->ClusterCount)Cache->ClusterCount+=num_clusters;
	return(0);
}

/* ****************************************************************************  
 * short fat_unlinkClusterChain(FileSystem *fs,ClusterChain *Cache)
 * Description: This function removes a clusterchain. Starting at FirstCluster
 * it follows the chain until the end, resetting all values to 0.
 * Return value: 0 on success.
*/
short fat_unlinkClusterChain(FileSystem *fs,ClusterChain *Cache)
{
	unsigned long c,tbd=0;
	
	Cache->LogicCluster=0;
	Cache->DiscCluster=Cache->FirstCluster;
	
	c=0;
	
	while(!fat_LogicToDiscCluster(fs,Cache,c++)){
		if(tbd!=0){
			fat_setNextClusterAddress(fs,tbd,0);
		}
		tbd=Cache->DiscCluster;
	}
	fat_setNextClusterAddress(fs,Cache->DiscCluster,0);
	fs->FreeClusterCount+=c;	
 	return(0);
}

unsigned long fat_countClustersInChain(FileSystem *fs,unsigned long firstcluster)
{
	ClusterChain cache;
	unsigned long c=0;
	
	if(firstcluster<=1)return(0);
	
	cache.DiscCluster = cache.LogicCluster = cache.LastCluster = cache.Linear = 0;
	cache.FirstCluster = firstcluster;
	
	while(!(fat_LogicToDiscCluster(fs,&cache,c++)));
	
	return(c-1);
}

unsigned long fat_DiscToLogicCluster(FileSystem *fs,unsigned long firstcluster,unsigned long disccluster)
{
	ClusterChain cache;
	unsigned long c=0,r=0;
	
	cache.DiscCluster = cache.LogicCluster = cache.LastCluster = cache.Linear = 0;
	cache.FirstCluster = firstcluster;
	
	while(!(fat_LogicToDiscCluster(fs,&cache,c++)) && !r){
		if(cache.DiscCluster == disccluster){
			r = cache.LogicCluster;
		}
	}
	return(r);
}

unsigned long fat_countFreeClusters(FileSystem *fs)
{
	unsigned long c=2,fc=0;
	
	while(c<=fs->DataClusterCount+1){
		if(fat_getNextClusterAddress(fs,c,0)==0)fc++;
		c++;
	}
	return(fc);
}
