#ifndef __TIME_H_
#define __TIME_H_

#ifdef DATE_TIME_SUPPORT
	#define time_getYear(void) efsl_getYear()
	#define time_getMonth(void) efsl_getMonth()
	#define time_getDay(void) efsl_getDay()
	#define time_getHour(void) efsl_getHour()
	#define time_getMinute(void) efsl_getMinute()
	#define time_getSecond(void) efsl_getSecond()
	#define time_getDate(void) fs_makeDate()
	#define time_getTime(void) fs_makeTime()
#else
	#define time_getYear(void) 0x0;
	#define time_getMonth(void) 0x0;
	#define time_getDay(void) 0x0;
	#define time_getHour(void) 0x0;
	#define time_getMinute(void) 0x0;
	#define time_getSecond(void) 0x0;
	#define time_getDate(void) 0x0;
	#define time_getTime(void) 0x0;
#endif

#ifdef DATE_TIME_SUPPORT
unsigned short efsl_getYear(void);
unsigned char  efsl_getMonth(void);
unsigned char  efsl_getDay(void);
unsigned char  efsl_getHour(void);
unsigned char  efsl_getMinute(void);
unsigned char  efsl_getSecond(void);
unsigned short fs_makeDate(void);
unsigned short fs_makeTime(void);
#endif

unsigned char fs_hasTimeSupport(void);

#endif
