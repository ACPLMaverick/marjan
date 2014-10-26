/*****************************************************************************/
#include "ioman.h"
/*****************************************************************************/

signed char ioman_init(IOManager *ioman, hwInterface *iface, unsigned char* bufferarea)
{
	ioman->iface=iface;
	
	ioman->bufptr = ioman_getBuffer(ioman,bufferarea);
	ioman->numbuf = IOMAN_NUMBUFFER;
	ioman->numit  = IOMAN_NUMITERATIONS;
	
	ioman_reset(ioman);
	return(0);
}
/*****************************************************************************/

void ioman_reset(IOManager *ioman)
{
	unsigned short nb,ni;
	
	memClr(ioman->sector,sizeof(unsigned long)*ioman->numbuf);
	memClr(ioman->status,sizeof(unsigned char) *ioman->numbuf);
	memClr(ioman->usage ,sizeof(unsigned char) *ioman->numbuf);
	memClr(ioman->itptr ,sizeof(unsigned char) *ioman->numbuf);
	ioman_setError(ioman,IOMAN_NOERROR);
		
	for(nb=0;nb<ioman->numbuf;nb++){
		for(ni=0;ni<ioman->numit;ni++){
			ioman->stack[nb][ni].sector=0;
			ioman->stack[nb][ni].status=0;
			ioman->stack[nb][ni].usage =0;
		}
	}
}
/*****************************************************************************/

unsigned char* ioman_getBuffer(IOManager *ioman,unsigned char* bufferarea)
{
#ifdef IOMAN_DO_MEMALLOC
	return(ioman->cache_mem);
#else
	return(bufferarea);
#endif
}
/*****************************************************************************/

void ioman_setAttr(IOManager *ioman,unsigned short bufplace,unsigned char attribute,unsigned char val)
{
	if(bufplace>=ioman->numbuf){
		ioman_setError(ioman,IOMAN_ERR_SETATTROUTOFBOUNDS);
		return; /* Out of bounds */
	}
	
	if(val){
		ioman->status[bufplace]|=1<<attribute;
	}else{
		ioman->status[bufplace]&=~(1<<attribute);
	}
}
/*****************************************************************************/

unsigned char ioman_getAttr(IOManager *ioman,unsigned short bufplace,unsigned char attribute)
{
	if(bufplace>=ioman->numbuf){
		ioman_setError(ioman,IOMAN_ERR_GETATTROUTOFBOUNDS);
		return(0xFF); /* Out of bounds */
	}

	return(ioman->status[bufplace]&(1<<attribute));
}
/*****************************************************************************/

unsigned char ioman_getUseCnt(IOManager *ioman,unsigned short bufplace)
{
	if(bufplace>=ioman->numbuf){
		ioman_setError(ioman,IOMAN_ERR_OPOUTOFBOUNDS);
		return(0x00);
	}
	return(ioman->usage[bufplace]);
}
/*****************************************************************************/


void ioman_incUseCnt(IOManager *ioman,unsigned short bufplace)
{
	if(bufplace>=ioman->numbuf){
		ioman_setError(ioman,IOMAN_ERR_OPOUTOFBOUNDS);
		return;
	}
	if(ioman->usage[bufplace]==0xFF)return;
	else ioman->usage[bufplace]++;
}
/*****************************************************************************/

void ioman_decUseCnt(IOManager *ioman,unsigned short bufplace)
{
	if(bufplace>=ioman->numbuf){
		ioman_setError(ioman,IOMAN_ERR_OPOUTOFBOUNDS);
		return;
	}
	if(ioman->usage[bufplace]==0x0)return;
	else ioman->usage[bufplace]--;
}
/*****************************************************************************/

void ioman_resetUseCnt(IOManager *ioman,unsigned short bufplace)
{
	if(bufplace>=ioman->numbuf){
		ioman_setError(ioman,IOMAN_ERR_OPOUTOFBOUNDS);
		return;
	}
	ioman->usage[bufplace]=0x00;
}
/*****************************************************************************/

unsigned char ioman_getRefCnt(IOManager *ioman,unsigned short bufplace)
{
	if(bufplace>=ioman->numbuf){
		ioman_setError(ioman,IOMAN_ERR_OPOUTOFBOUNDS);
		return(0x00);
	}
	return(ioman->reference[bufplace]);
}
/*****************************************************************************/

void ioman_incRefCnt(IOManager *ioman,unsigned short bufplace)
{
	if(bufplace>=ioman->numbuf){
		ioman_setError(ioman,IOMAN_ERR_OPOUTOFBOUNDS);
		return;
	}
	if(ioman->reference[bufplace]==0xFF)return;
	else ioman->reference[bufplace]++;
}
/*****************************************************************************/

void ioman_decRefCnt(IOManager *ioman,unsigned short bufplace)
{
	if(bufplace>=ioman->numbuf){
		ioman_setError(ioman,IOMAN_ERR_OPOUTOFBOUNDS);
		return;
	}
	if(ioman->reference[bufplace]==0x00)return;
	else ioman->reference[bufplace]--;
}
/*****************************************************************************/

void ioman_resetRefCnt(IOManager *ioman,unsigned short bufplace)
{
	if(bufplace>=ioman->numbuf){
		ioman_setError(ioman,IOMAN_ERR_OPOUTOFBOUNDS);
		return;
	}
	ioman->reference[bufplace]=0x00;
}
/*****************************************************************************/

signed char ioman_pop(IOManager *ioman,unsigned short bufplace)
{
	if(bufplace>=ioman->numbuf){
		ioman_setError(ioman,IOMAN_ERR_POPEMPTYSTACK);
		return(-1);
	}
	if(ioman->itptr[bufplace]==0 || ioman->itptr[bufplace]>IOMAN_NUMITERATIONS)return(-1);
	ioman->sector[bufplace] = ioman->stack[bufplace][ioman->itptr[bufplace]].sector;
	ioman->status[bufplace] = ioman->stack[bufplace][ioman->itptr[bufplace]].status;
	ioman->usage[bufplace]  = ioman->stack[bufplace][ioman->itptr[bufplace]].usage; 
	ioman->itptr[bufplace]--;
	return(0);
}
/*****************************************************************************/

signed char ioman_push(IOManager *ioman,unsigned short bufplace)
{
	if(bufplace>=ioman->numbuf){
		ioman_setError(ioman,IOMAN_ERR_OPOUTOFBOUNDS);
		return(-1);
	}
	if(ioman->itptr[bufplace]>=IOMAN_NUMITERATIONS){
		ioman_setError(ioman,IOMAN_ERR_PUSHBEYONDSTACK);	
		return(-1);
	}
	ioman->itptr[bufplace]++;
	ioman->stack[bufplace][ioman->itptr[bufplace]].sector = ioman->sector[bufplace];
	ioman->stack[bufplace][ioman->itptr[bufplace]].status = ioman->status[bufplace];
	ioman->stack[bufplace][ioman->itptr[bufplace]].usage  = ioman->usage[bufplace];
	return(0);
}
/*****************************************************************************/

unsigned char* ioman_getPtr(IOManager *ioman,unsigned short bufplace)
{
	if(bufplace>=ioman->numbuf){
		ioman_setError(ioman,IOMAN_ERR_OPOUTOFBOUNDS);
		return(0);
	}
	return(ioman->bufptr+bufplace*512);
}
/*****************************************************************************/

signed short ioman_getBp(IOManager *ioman,unsigned char* buf)
{
	if(buf<(ioman->bufptr) || buf>=( ioman->bufptr+(ioman->numbuf*512) )){
		ioman_setError(ioman,IOMAN_ERR_CACHEPTROUTOFRANGE);
		return(-1);
	}
	return((buf-(ioman->bufptr))/512);
}
/*****************************************************************************/

signed char ioman_readSector(IOManager *ioman,unsigned long address,unsigned char* buf)
{
	signed char r;

	if(buf==0){
		return(-1);
	}
	
	r=if_readBuf(ioman->iface,address,buf);
	
	if(r!=0){
		ioman_setError(ioman,IOMAN_ERR_READFAIL);
		return(-1);
	}
	return(0);
}
/*****************************************************************************/

signed char ioman_writeSector(IOManager *ioman, unsigned long address, unsigned char* buf)
{
	signed char r;

	if(buf==0)return(-1);
	
	r=if_writeBuf(ioman->iface,address,buf);

	if(r<=0){
		ioman_setError(ioman,IOMAN_ERR_WRITEFAIL);
		return(-1);
	}
	return(0);
}
/*****************************************************************************/

void ioman_resetCacheItem(IOManager *ioman,unsigned short bufplace)
{
	if(bufplace>=ioman->numbuf){
		ioman_setError(ioman,IOMAN_ERR_OPOUTOFBOUNDS);
		return;
	}
	ioman->sector[bufplace]    = 0;
	ioman->status[bufplace]    = 0;
	ioman->usage[bufplace]     = 0;
	ioman->reference[bufplace] = 0;
}
/*****************************************************************************/

signed long ioman_findSectorInCache(IOManager *ioman, unsigned long address)
{
	unsigned short c;
	
	for(c=0;c<ioman->numbuf;c++){
		if(ioman_isValid(c) && ioman->sector[c] == address)return(c);
	}
	return(-1);
}
/*****************************************************************************/

signed long ioman_findFreeSpot(IOManager *ioman)
{
	unsigned short c;
	
	for(c=0;c<ioman->numbuf;c++){
		if(!ioman_isValid(c))return(c);
	}
	return(-1);
}
/*****************************************************************************/

signed long ioman_findUnusedSpot(IOManager *ioman)
{
	signed long r=-1;
	unsigned short c;
	unsigned char fr=0,lr=0xFF;
	
	for(c=0;c<ioman->numbuf;c++){
		if(ioman_getUseCnt(ioman,c)==0){
			if(!ioman_isWritable(c) && !fr){
				fr=1;
				lr=0xFF;
				r=-1;
			}
			if(ioman_isWritable(c) && !fr){
				if(ioman_getRefCnt(ioman,c)<=lr){
					r=c;
					lr=ioman_getRefCnt(ioman,c);
				}
			}
			if(fr && !ioman_isWritable(c)){
				if(ioman_getRefCnt(ioman,c)<=lr){
					r=c;
					lr=ioman_getRefCnt(ioman,c);
				}
			}
		}
	}
	return(r);
}
/*****************************************************************************/

signed long ioman_findOverallocableSpot(IOManager *ioman)
{
	unsigned char points,lp=0xFF;
	unsigned short c;
	signed long r=-1;
	
	for(c=0;c<ioman->numbuf;c++){
		if(ioman->itptr[c]<ioman->numit){
			points = 0;
			if(ioman_isWritable(c))points+=0x7F;
			points += ((unsigned short)(ioman->itptr[c]*0x4D))/(ioman->numit);
			points += ((unsigned short)(ioman_getRefCnt(ioman,c)*0x33))/0xFF;
			if(points<lp){
				lp=points;
				r=c;
			}
		}
	}
	return(r);
}
/*****************************************************************************/

signed char ioman_putSectorInCache(IOManager *ioman, unsigned long address, unsigned short bufplace)
{
	unsigned char* buf;
	
	if((buf = ioman_getPtr(ioman,bufplace))==0){
		ioman_setError(ioman,IOMAN_ERR_CACHEPTROUTOFRANGE);
		return(-1);
	}
	if((ioman_readSector(ioman,address,buf))){
		ioman_setError(ioman,IOMAN_ERR_READFAIL);
		return(-1);
	}
	ioman_setValid(bufplace);
	ioman->sector[bufplace]=address;
	return(0);
}
/*****************	if(bufplace>=ioman->numbuf)return;
************************************************************/

signed char ioman_flushSector(IOManager *ioman, unsigned short bufplace)
{
	unsigned char* buf;
	
	if((buf = ioman_getPtr(ioman,bufplace))==0){
		ioman_setError(ioman,IOMAN_ERR_CACHEPTROUTOFRANGE);
		return(-1);
	}
	if(!ioman_isWritable(bufplace)){
		ioman_setError(ioman,IOMAN_ERR_WRITEREADONLYSECTOR);
		return(-1);
	}
	if(!(ioman_writeSector(ioman,ioman->sector[bufplace],buf))){
		ioman_setError(ioman,IOMAN_ERR_WRITEFAIL);	
		return(-1);
	}
	if(ioman->usage==0)ioman_setNotWritable(bufplace);
	return(0);
}
/*****************************************************************************/

signed char ioman_flushRange(IOManager *ioman,unsigned long address_low, unsigned long address_high)
{
	unsigned long c;
	
	if(address_low>address_high){
		c=address_low; address_low=address_high;address_high=c;
	}
	
	for(c=0;c<ioman->numbuf;c++){
		if((ioman->sector[c]>=address_low) && (ioman->sector[c]<=address_high) && (ioman_isWritable(c))){
			if(ioman_flushSector(ioman,c)){
				return(-1);
			}
			if(ioman->usage[c]==0)ioman_setNotWritable(c);
		}
	}
	return(0);
}
/*****************************************************************************/

signed char ioman_flushAll(IOManager *ioman)
{
	unsigned short c;
	
	for(c=0;c<ioman->numbuf;c++){
		if(ioman_isWritable(c)){
			if(ioman_flushSector(ioman,c)){
				return(-1);
			}
			if(ioman->usage[c]==0)ioman_setNotWritable(c);
		}
	}
	return(0);
}
/*****************************************************************************/

unsigned char* ioman_getSector(IOManager *ioman,unsigned long address, unsigned char mode)
{
	signed long bp;
	
	if((bp=ioman_findSectorInCache(ioman,address))!=-1){
		if(ioman_isReqRw(mode)){
			ioman_setWritable(bp);
		}
		ioman_incUseCnt(ioman,bp);
		if(!ioman_isReqExp(mode))ioman_incRefCnt(ioman,bp);
		return(ioman_getPtr(ioman,bp));
	}
	
	if((bp=ioman_findFreeSpot(ioman))==-1){
		if(((bp=ioman_findUnusedSpot(ioman))!=-1)&&(ioman_isWritable(bp))){
			ioman_flushSector(ioman,bp);
		}
	}
	
	if(bp!=-1){
		ioman_resetCacheItem(ioman,bp);
		if((ioman_putSectorInCache(ioman,address,bp))){
			return(0);
		}
		if(mode==IOM_MODE_READWRITE){
			ioman_setWritable(bp);
		}
		ioman_incUseCnt(ioman,bp);
		if(!ioman_isReqExp(mode))ioman_incRefCnt(ioman,bp);
		return(ioman_getPtr(ioman,bp));
	}
	
	if((bp=ioman_findOverallocableSpot(ioman))!=-1){
		if(ioman_isWritable(bp)){
			ioman_flushSector(ioman,bp);
		}
		if(ioman_push(ioman,bp)){
			return(0);
		}
		ioman_resetCacheItem(ioman,bp);
		if((ioman_putSectorInCache(ioman,address,bp))){
			return(0);
		}
		if(ioman_isReqRw(mode)){
			ioman_setWritable(bp);
		}
		ioman_incUseCnt(ioman,bp);
		if(!ioman_isReqExp(mode))ioman_incRefCnt(ioman,bp);
		return(ioman_getPtr(ioman,bp));
	}
	ioman_setError(ioman,IOMAN_ERR_NOMEMORY);
	return(0);
}
/*****************************************************************************/

signed char ioman_releaseSector(IOManager *ioman,unsigned char* buf)
{
	unsigned short bp;
	
	bp=ioman_getBp(ioman,buf);
	ioman_decUseCnt(ioman,bp);
	
	if(ioman_getUseCnt(ioman,bp)==0 && ioman->itptr[bp]!=0){
		if(ioman_isWritable(bp)){
			ioman_flushSector(ioman,bp);
		}
		ioman_pop(ioman,bp);
		ioman_putSectorInCache(ioman,ioman->sector[bp],bp);
	}
	return(0);
}
/*****************************************************************************/

signed char ioman_directSectorRead(IOManager *ioman,unsigned long address, unsigned char* buf)
{
	unsigned char* ibuf;
	signed short bp;
	
	if((bp=ioman_findSectorInCache(ioman,address))!=-1){
		ibuf=ioman_getPtr(ioman,bp);
		memCpy(ibuf,buf,512);
		return(0);
	}
	
	if((bp=ioman_findFreeSpot(ioman))!=-1){
		if((ioman_putSectorInCache(ioman,address,bp))){
			return(-1);
		}
		ibuf=ioman_getPtr(ioman,bp);
		memCpy(ibuf,buf,512);
		return(0);
	}

	if(ioman_readSector(ioman,address,buf)){
		return(-1);
	}

	return(0);
}
/*****************************************************************************/

signed char ioman_directSectorWrite(IOManager *ioman,unsigned long address, unsigned char* buf)
{
	unsigned char* ibuf;
	signed short bp;
	
	if((bp=ioman_findSectorInCache(ioman,address))!=-1){
		ibuf=ioman_getPtr(ioman,bp);
		memCpy(buf,ibuf,512);
		ioman_setWritable(bp);
		return(0);
	}
	
	if((bp=ioman_findFreeSpot(ioman))!=-1){
		ibuf=ioman_getPtr(ioman,bp);
		memCpy(buf,ibuf,512);
		ioman_resetCacheItem(ioman,bp);
		ioman->sector[bp]=address;
		ioman_setWritable(bp);
		ioman_setValid(bp);
		return(0);
	}

	if(ioman_writeSector(ioman,address,buf)){
		return(-1);
	}

	return(0);
}
/*****************************************************************************/

void ioman_printStatus(IOManager *ioman)
{
	unsigned short c;
	
	DBG((TXT("IO-Manager -- Report\n====================\n")));
	DBG((TXT("Buffer is %i sectors, from %p to %p\n"),
	          ioman->numbuf,ioman->bufptr,ioman->bufptr+(ioman->numbuf*512)));
	for(c=0;c<ioman->numbuf;c++){
		if(ioman_isValid(c)){
			DBG((TXT("BP %3i\t SC %8li\t\t US %i\t RF %i\t %s %s\n"),
				c,ioman->sector[c],ioman_getUseCnt(ioman,c),ioman_getRefCnt(ioman,c),
				ioman_isUserBuf(c) ? "USRBUF" : "      ",
				ioman_isWritable(c) ? "WRITABLE" : "READONLY"));
		}
	}
}
/*****************************************************************************/

