#ifndef __FAT_H_
#define __FAT_H_

/*****************************************************************************/
#include "config.h"
#include "error.h"
#include "file.h"
/*****************************************************************************/

unsigned long fat_getSectorAddressFatEntry(FileSystem *fs,unsigned long cluster_addr);
unsigned long fat_getNextClusterAddress(FileSystem *fs,	unsigned long cluster_addr, unsigned short *linear);
void fat_setNextClusterAddress(FileSystem *fs,unsigned long cluster_addr,unsigned long next_cluster_addr);
short fat_isEocMarker(FileSystem *fs,unsigned long fat_entry);
unsigned long fat_giveEocMarker(FileSystem *fs);
unsigned long fat_findClusterAddress(FileSystem *fs,unsigned long cluster,unsigned long offset, unsigned char *linear);
unsigned long fat_getNextClusterAddressWBuf(FileSystem *fs,unsigned long cluster_addr, unsigned char * buf);
void fat_setNextClusterAddressWBuf(FileSystem *fs,unsigned long cluster_addr,unsigned long next_cluster_addr,unsigned char * buf);
signed short fat_getNextClusterChain(FileSystem *fs, ClusterChain *Cache);
void fat_bogus(void);
signed short fat_LogicToDiscCluster(FileSystem *fs, ClusterChain *Cache,unsigned long logiccluster);
short fat_allocClusterChain(FileSystem *fs,ClusterChain *Cache,unsigned long num_clusters);
short fat_unlinkClusterChain(FileSystem *fs,ClusterChain *Cache);
unsigned long fat_countClustersInChain(FileSystem *fs,unsigned long firstcluster);
unsigned long fat_DiscToLogicCluster(FileSystem *fs,unsigned long firstcluster,unsigned long disccluster);
unsigned long fat_countFreeClusters(FileSystem *fs);

#endif
