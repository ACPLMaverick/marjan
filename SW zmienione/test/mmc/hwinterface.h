#ifndef __HWINTERFACE_H_ 
#define __HWINTERFACE_H_ 

#ifndef FALSE
#define FALSE	0x00
#define TRUE	0x01
#endif

#include "config.h"


/*************************************************************\
              hwInterface
               ----------
* FILE* 	imagefile		File emulation of hw interface.
* long		sectorCount		Number of sectors on the file.
\*************************************************************/
struct  hwInterface{
	/*FILE 	*imageFile;*/
	long 	sectorCount;
};
typedef struct hwInterface hwInterface;

signed char if_initInterface(hwInterface* file, char* opts);
signed char if_readBuf(hwInterface* file, unsigned long address, unsigned char* buf);
signed char if_writeBuf(hwInterface* file, unsigned long address, unsigned char* buf);
signed char if_setPos(hwInterface* file, unsigned long address);

void if_spiInit(hwInterface *iface);
void if_spiSetSpeed(unsigned char speed);
unsigned char if_spiSend(hwInterface *iface, unsigned char outgoing);

#endif
