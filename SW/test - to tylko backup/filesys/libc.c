/*****************************************************************************/
#include "libc.h"
/*****************************************************************************/

/* ****************************************************************************  
 * unsigned short strMatch(char* bufa, char*bufb, unsigned long n)
 * Description: Compares bufa and bufb for a length of n bytes.
 * Return value: Returns the number of character NOT matching.
*/
unsigned short strMatch(char* bufa, char*bufb,unsigned long n)
{
	unsigned long c;
	unsigned short res=0;
	for(c=0;c<n;c++)if(bufa[c]!=bufb[c])res++;
	return(res);
}
/*****************************************************************************/ 


/* ****************************************************************************  
 * void memCpy(void* psrc, void* pdest, unsigned long size)
 * Description: Copies the contents of psrc into pdest on a byte per byte basis.
 * The total number of bytes copies is size.
*/
void memCpy(void* psrc, void* pdest, unsigned long size)
{
	while(size>0){
		*((char*)pdest+size-1)=*((char*)psrc+size-1);
		size--;
	}
}
/*****************************************************************************/ 

void memClr(void *pdest,unsigned long size)
{
	while(size>0){
		*(((char*)pdest)+size-1)=0x00;
		size--;
	}
}

void memSet(void *pdest,unsigned long size,unsigned char data)
{
	while(size>0){
		*(((char*)pdest)+size-1)=data;
		size--;
	}
}


