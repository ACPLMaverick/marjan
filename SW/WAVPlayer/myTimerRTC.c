#include <lpc2xxx.h>
#include "mp3shared.h"
#include "wavplayer.h"
#include "myTimerRTC.h"

SongInfo currentSongInfo;

extern void ISR(void);

void myTimerExec(void);
void setupPLL(void);
void connectPLL(void);
void initClocks(void);
void initTimer1(void);
void initRTC(void);
void setupInterrupts(void);
void T1ISR(void);
void RTCISR(void);
void feedSeq(void);

void myTimerRTCExec(void)
{
	initClocks();
	initTimer1();
	initRTC();
	setupInterrupts();

	T1TCR = 0x01;
}

void setupPLL(void)
{
	PLLCON = 0x01;	// PPLE = 1, PPLC = 0, czyli Enabled ale nie Connected
	PLLCFG = 0x24;	// M = 5 (4), P = 2 (01)
}

void connectPLL(void)
{
	while(!(PLLSTAT & PLOCK));
	PLLCON = 0x03;	// PPLE = 1, PPLC = 1
}

void feedSeq(void)
{
	PLLFEED = PLLFEEDCODE01;
	PLLFEED = PLLFEEDCODE02;
}

void initClocks(void)
{
	// PLL setup
	setupPLL();
	feedSeq();
	connectPLL();
	feedSeq();

	//VPBDIV = 0x01; -already set
	//////
}

void initTimer1(void)
{
	// konfiguracja timera
	T1CTCR = 0x0;
	T1PR = PRESCALE - 1; // ustalam po ilu tickach zegara TC ma byc inkrementowany
	T1MR0 = DELAY_MS - 1;
	T1MCR = MR0I | MR0R; // czyli bitowo 11, Interrupt, Reset TC on MR0

	T1TCR = 0x02;	// reset timera
}

void initRTC(void)
{
	RTC_ILR = (1 << 0) | (1 << 1);
	RTC_CCR = (1 << 0) | (1 << 4);	// w³¹czenie zegara, ustawienie by korzysta³ z 32kHz oscylatora
	RTC_SEC = RTC_MIN = RTC_HOUR = 0;
	RTC_CIIR = (1 << 0);
}

void setupInterrupts(void)
{
	// konfiguracja przerwania
	VICVectAddr5 = (unsigned )T1ISR;	// podpiêcie adresu funkcji do rejestru
	VICVectCntl5 = (1 << 5) | 5;	// 5 bit - w³¹czenie Vectored IRQ
									// 0x5 - numer Ÿród³owy Timera 1, którego channel mask wynosi w³aœnie 5

	// dla RTC podobnie
	VICVectAddr13 = (unsigned )RTCISR;
	VICVectCntl13 = (1 << 5) | 13;

	VICIntEnable = (1 << 5) | (1 << 13);
}

void T1ISR(void)
{
	long int regVal;
	regVal = T1IR;
	/////////////////

	ISR();

	////////////////
	T1IR = regVal;		// czyœcimy Interrupt Flag
	VICVectAddr = 0x0;	// sygna³ koñca przerwania
}

void RTCISR(void)
{
	ISR_RTC();
	RTC_ILR |= (1 << 0);
	VICVectAddr = 0x0;
}

void StopInterrupts(void)
{
	VICVectAddr5 = 0x00;
	VICVectCntl5 = 0x00;
	VICVectAddr13 = 0x0;
	VICVectCntl13 = 0x0;
}


