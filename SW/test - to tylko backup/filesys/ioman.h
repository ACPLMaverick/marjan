#ifndef __IOMAN_H__
#define __IOMAN_H__

/*****************************************************************************/
#include "../mmc/interface.h"
#include "error.h"
#include "libc.h"
#include "config.h"
/*****************************************************************************/

#define IOMAN_STATUS_ATTR_VALIDDATA  0
#define IOMAN_STATUS_ATTR_USERBUFFER 1
#define IOMAN_STATUS_ATTR_WRITE      2

#define IOM_MODE_READONLY  1
#define IOM_MODE_READWRITE 2
#define IOM_MODE_EXP_REQ   4

struct IOManStack{
	unsigned long sector;
	unsigned char  status;
	unsigned char  usage;
};
typedef struct IOManStack IOManStack;

struct IOManager{
	hwInterface *iface;
	
	unsigned char *bufptr;
	unsigned short numbuf;
	unsigned short numit;
		
	IOMAN_ERR_EUINT8
		
	IOManStack stack[IOMAN_NUMBUFFER][IOMAN_NUMITERATIONS];
	
	unsigned long sector[IOMAN_NUMBUFFER];
	unsigned char  status[IOMAN_NUMBUFFER];
	unsigned char  usage[IOMAN_NUMBUFFER];
	unsigned char  reference[IOMAN_NUMBUFFER];
	unsigned char  itptr[IOMAN_NUMBUFFER];
#ifdef IOMAN_DO_MEMALLOC
	unsigned char  cache_mem[IOMAN_NUMBUFFER * 512];
#endif
};
typedef struct IOManager IOManager;

#define IOBJ ioman

#define ioman_isValid(bp) ioman_getAttr(IOBJ,bp,IOMAN_STATUS_ATTR_VALIDDATA)
#define ioman_isUserBuf(bp) ioman_getAttr(IOBJ,bp,IOMAN_STATUS_ATTR_USERBUFFER)
#define ioman_isWritable(bp) ioman_getAttr(IOBJ,bp,IOMAN_STATUS_ATTR_WRITE)

#define ioman_setValid(bp) ioman_setAttr(IOBJ,bp,IOMAN_STATUS_ATTR_VALIDDATA,1)
#define ioman_setUserBuf(bp) ioman_setAttr(IOBJ,bp,IOMAN_STATUS_ATTR_USERBUFFER,1)
#define ioman_setWritable(bp) ioman_setAttr(IOBJ,bp,IOMAN_STATUS_ATTR_WRITE,1)

#define ioman_setNotValid(bp) ioman_setAttr(IOBJ,bp,IOMAN_STATUS_ATTR_VALIDDATA,0)
#define ioman_setNotUserBuf(bp) ioman_setAttr(IOBJ,bp,IOMAN_STATUS_ATTR_USERBUFFER,0)
#define ioman_setNotWritable(bp) ioman_setAttr(IOBJ,bp,IOMAN_STATUS_ATTR_WRITE,0)

#define ioman_isReqRo(mode)  ((mode)&(IOM_MODE_READONLY))
#define ioman_isReqRw(mode)  ((mode)&(IOM_MODE_READWRITE))
#define ioman_isReqExp(mode) ((mode)&(IOM_MODE_EXP_REQ))

signed char ioman_init(IOManager *ioman, hwInterface *iface, unsigned char* bufferarea);
void ioman_reset(IOManager *ioman);
unsigned char* ioman_getBuffer(IOManager *ioman,unsigned char* bufferarea);
void ioman_setAttr(IOManager *ioman,unsigned short bufplace,unsigned char attribute,unsigned char val);
unsigned char ioman_getAttr(IOManager *ioman,unsigned short bufplace,unsigned char attribute);
unsigned char ioman_getUseCnt(IOManager *ioman,unsigned short bufplace);
void ioman_incUseCnt(IOManager *ioman,unsigned short bufplace);
void ioman_decUseCnt(IOManager *ioman,unsigned short bufplace);
void ioman_resetUseCnt(IOManager *ioman,unsigned short bufplace);
unsigned char ioman_getRefCnt(IOManager *ioman,unsigned short bufplace);
void ioman_incRefCnt(IOManager *ioman,unsigned short bufplace);
void ioman_decRefCnt(IOManager *ioman,unsigned short bufplace);
void ioman_resetRefCnt(IOManager *ioman,unsigned short bufplace);
signed char ioman_pop(IOManager *ioman,unsigned short bufplace);
signed char ioman_push(IOManager *ioman,unsigned short bufplace);
unsigned char* ioman_getPtr(IOManager *ioman,unsigned short bufplace);
signed short ioman_getBp(IOManager *ioman,unsigned char* buf);
signed char ioman_readSector(IOManager *ioman,unsigned long address,unsigned char* buf);
signed char ioman_writeSector(IOManager *ioman, unsigned long address, unsigned char* buf);
void ioman_resetCacheItem(IOManager *ioman,unsigned short bufplace);
signed long ioman_findSectorInCache(IOManager *ioman, unsigned long address);
signed long ioman_findFreeSpot(IOManager *ioman);
signed long ioman_findUnusedSpot(IOManager *ioman);
signed long ioman_findOverallocableSpot(IOManager *ioman);
signed char ioman_putSectorInCache(IOManager *ioman,unsigned long address, unsigned short bufplace);
signed char ioman_flushSector(IOManager *ioman, unsigned short bufplace);
unsigned char* ioman_getSector(IOManager *ioman,unsigned long address, unsigned char mode);
signed char ioman_releaseSector(IOManager *ioman,unsigned char* buf);
signed char ioman_directSectorRead(IOManager *ioman,unsigned long address, unsigned char* buf);
signed char ioman_directSectorWrite(IOManager *ioman,unsigned long address, unsigned char* buf);
signed char ioman_flushRange(IOManager *ioman,unsigned long address_low, unsigned long address_high);
signed char ioman_flushAll(IOManager *ioman);

void ioman_printStatus(IOManager *ioman);

#endif
