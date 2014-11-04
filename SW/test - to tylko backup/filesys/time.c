/*****************************************************************************/
#include "time.h"
/*****************************************************************************/

unsigned short fs_makeDate(void)
{
#ifndef DATE_TIME_SUPPORT
	return(0);
#else
	unsigned char m,d;
	unsigned short y;
	
	y = time_getYear()-1980;
	m = time_getMonth();
	d = time_getDay();
	
	return(
		(y>127?127<<9:(y&0x3F)<<9)   |
		((m==0||m>12)?1:(m&0xF)<<5)  |
		((d==0||d>31)?1:(d&0x1F))
	);
#endif
}
/*****************************************************************************/

unsigned short fs_makeTime(void)
{
#ifndef DATE_TIME_SUPPORT
	return(0);
#else
	unsigned char s,m,h;
	
	s = time_getSecond();
	m = time_getMinute();
	h = time_getHour();
	
	return(
		(h>23?0:(h&0x1F)<<11) |
		(m>59?0:(m&0x3F)<<5)  |
		(s>59?0:(s-s%2)/2)
	);
#endif
}
/*****************************************************************************/

unsigned char fs_hasTimeSupport(void)
{
#ifdef DATE_TIME_SUPPORT
	return(1);
#else
	return(0);
#endif
}
/*****************************************************************************/


