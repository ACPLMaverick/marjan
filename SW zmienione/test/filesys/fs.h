#ifndef __FS_H_
#define __FS_H_

/*****************************************************************************/
#include "config.h"
#include "error.h"
#include "partition.h"
#include "time.h"
/*****************************************************************************/

#define FAT12 1
#define FAT16 2
#define FAT32 3

#define FS_INFO_SECTOR 1
#define FSINFO_MAGIC_BEGIN 0x41615252
#define FSINFO_MAGIC_END   0xAA550000

struct VolumeId{
	unsigned short BytesPerSector;			/*Must be 512*/
	unsigned char SectorsPerCluster;		/*Must be multiple of 2 (1,2,4,8,16 or 32)*/
	unsigned short ReservedSectorCount;	/*Number of sectors after which the first FAT begins.*/
	unsigned char NumberOfFats;			/*Should be 2*/
	unsigned short RootEntryCount;			/*Number of filerecords the Rootdir can contain. NOT for FAT32*/
	unsigned short SectorCount16;			/*Number of Sectors for 12/16 bit FAT */
	unsigned short FatSectorCount16;		/*Number of Sectors for 1 FAT on FAT12/16 bit FAT's*/
	unsigned long SectorCount32;			/*Number of Sectors for 32 bit FAT*/
	unsigned long FatSectorCount32;		/*Number of Sectors for 1 FAT on FAT32 */
	unsigned long RootCluster;			/*Clusternumber of the first cluster of the RootDir on FAT 32*/
};
typedef struct VolumeId VolumeId;

struct FileSystem{
	Partition *part;					/*Pointer to partition on which this FS resides.*/
	VolumeId volumeId;					/*Contains important FS info.*/
	unsigned long DataClusterCount;			/*Number of dataclusters. This number determines the FATType.*/
	unsigned long FatSectorCount;				/*Number of sectors for 1 FAT, regardless of FATType*/
	unsigned long SectorCount;				/*Number of sectors, regardless of FATType*/
	unsigned long FirstSectorRootDir;			/*First sector of the RootDir. */
	unsigned long FirstClusterCurrentDir;
	unsigned long FreeClusterCount;
	unsigned long NextFreeCluster;
	unsigned char type;						/*Determines FATType (FAT12 FAT16 or FAT32 are defined)*/
};
typedef struct FileSystem FileSystem;

struct FileLocation{
	unsigned long Sector;					/*Sector where the directoryentry of the file/directory can be found.*/
	unsigned char Offset;					/*Offset (in 32byte segments) where in the sector the entry is.*/
	unsigned char attrib;
};
typedef struct FileLocation FileLocation;

struct ClusterChain{
	unsigned char Linear;				/*For how many more clusters the file is nonfragmented*/
	signed long LogicCluster;		/*This field determines the n'th cluster of the file as current*/
	unsigned long DiscCluster;		/*If this field is 0, it means the cache is invalid.*/
	unsigned long FirstCluster;		/*First cluster of the chain. Zero or one are invalid.*/
	unsigned long LastCluster;		/*Last cluster of the chain (is not always filled in)*/
	unsigned long ClusterCount;
};
typedef struct ClusterChain ClusterChain;

struct FileRecord{
	unsigned char FileName[11];
	unsigned char Attribute;
	unsigned char NTReserved;
	unsigned char MilliSecTimeStamp;
	unsigned short CreatedTime;
	unsigned short CreatedDate;
	unsigned short AccessDate;
	unsigned short FirstClusterHigh;
	unsigned short WriteTime;
	unsigned short WriteDate;
	unsigned short FirstClusterLow;
	unsigned long FileSize;
};
typedef struct FileRecord FileRecord;


short fs_initFs(FileSystem *fs,Partition *part);
short fs_isValidFat(Partition *part);
void fs_loadVolumeId(FileSystem *fs, Partition *part);
signed short fs_verifySanity(FileSystem *fs);
void fs_countDataSectors(FileSystem *fs);
void fs_determineFatType(FileSystem *fs);
void fs_findFirstSectorRootDir(FileSystem *fs);
void fs_initCurrentDir(FileSystem *fs);
unsigned long fs_getSectorAddressRootDir(FileSystem *fs,unsigned long secref);
unsigned long fs_clusterToSector(FileSystem *fs,unsigned long cluster);
unsigned long fs_sectorToCluster(FileSystem *fs,unsigned long sector);
unsigned long fs_getNextFreeCluster(FileSystem *fs,unsigned long startingcluster);
unsigned long fs_giveFreeClusterHint(FileSystem *fs);
signed short fs_findFreeFile(FileSystem *fs,char* filename,FileLocation *loc,unsigned char mode);
signed char fs_findFile(FileSystem *fs,char* filename,FileLocation *loc,unsigned long *lastDir);
signed char fs_findFile_broken(FileSystem *fs,char* filename,FileLocation *loc);
unsigned long fs_getLastCluster(FileSystem *fs,ClusterChain *Cache);
unsigned long fs_getFirstClusterRootDir(FileSystem *fs);
unsigned short fs_makeDate(void);
unsigned short fs_makeTime(void);
void fs_setFirstClusterInDirEntry(FileRecord *rec,unsigned long cluster_addr);
void fs_initClusterChain(FileSystem *fs,ClusterChain *cache,unsigned long cluster_addr);
signed char fs_flushFs(FileSystem *fs);
signed char fs_umount(FileSystem *fs);
signed char fs_clearCluster(FileSystem *fs,unsigned long cluster);
signed char fs_getFsInfo(FileSystem *fs,unsigned char force_update);
signed char fs_setFsInfo(FileSystem *fs);

#endif
