#ifndef __LIBC_H__
#define __LIBC_H__

/*****************************************************************************/
#include "config.h"
/*****************************************************************************/

unsigned short strMatch(char* bufa, char* bufb,unsigned long n);
void memCpy(void* psrc, void* pdest, unsigned long size);
void memClr(void *pdest,unsigned long size);
void memSet(void *pdest,unsigned long size,unsigned char data);


#endif
