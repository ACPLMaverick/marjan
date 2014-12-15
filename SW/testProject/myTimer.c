#include <lpc2xxx.h>
#include "mp3shared.h"

#define PLOCK (0x00000500)	// lock bit of PLLSTAT
#define MR0I (1 << 0) // przerwanie wywo³ane kiedy TC == MR0
#define MR0R (1 << 1) // resetowanie TC kiedy TC == MR0
#define DELAY_MS 1000
#define PRESCALE 60000	// TC inkrementowane co 60 000 cykli zegara

#define PLLFEEDCODE01 0xAA
#define PLLFEEDCODE02 0x55

SongInfo currentSongInfo;

void myTimerExec(void);
void setupPLL(void);
void connectPLL(void);
void initClocks(void);
void initTimer1(void);
void T1ISR(void);
void feedSeq(void);

void myTimerExec(void)
{
	initClocks();
	initTimer1();

	T1TCR = 0x01;

	for(;;);
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

	// konfiguracja przerwania
	VICVectAddr5 = (unsigned )T1ISR;	// podpiêcie adresu funkcji do rejestru
	VICVectCntl5 = 0x20 | 5;	// 0x20 - w³¹czenie Vectored IRQ
								// 0x4 - numer Ÿród³owy Timera 1, którego channel mask wynosi w³aœnie 5
	VICIntEnable = 0x20;
	T1TCR = 0x02;	// reset timera
}

void T1ISR(void)
{
	long int regVal;
	regVal = T1IR;
	/////////////////

	currentSongInfo.time += 1;

	////////////////
	T1IR = regVal;		// czyœcimy Interrupt Flag
	VICVectAddr = 0x0;	// sygna³ koñca przerwania
}


