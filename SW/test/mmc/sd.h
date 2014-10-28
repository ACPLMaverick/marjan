
#ifndef __SD_H_ 
#define __SD_H_ 

#include "config.h"
#include "types.h"
#include "debug.h"

#ifdef HW_ENDPOINT_LPC2000_SD
	#include "hwinterface.h"
#endif

#define	CMDREAD		17
#define	CMDWRITE	24
#define	CMDREADCSD       9

signed char  sd_Init(hwInterface *iface);
void sd_Command(hwInterface *iface,unsigned char cmd, unsigned short paramx, unsigned short paramy);
unsigned char sd_Resp8b(hwInterface *iface);
signed short sd_Resp8bError(hwInterface *iface,unsigned char value);
unsigned short sd_Resp16b(hwInterface *iface);
signed char sd_State(hwInterface *iface);

signed char sd_readSector(hwInterface *iface,unsigned long address,unsigned char* buf, unsigned short len);
signed char sd_writeSector(hwInterface *iface,unsigned long address, unsigned char* buf);
signed char sd_getDriveSize(hwInterface *iface, unsigned long* drive_size );

#endif
