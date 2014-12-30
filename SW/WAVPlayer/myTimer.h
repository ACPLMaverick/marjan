/*
 * myTimer.h
 *
 *  Created on: 2014-12-30
 *      Author: embedded
 */

#ifndef MYTIMER_H_
#define MYTIMER_H_

#define PLOCK (0x00000500)	// lock bit of PLLSTAT
#define MR0I (1 << 0) // przerwanie wywo³ane kiedy TC == MR0
#define MR0R (1 << 1) // resetowanie TC kiedy TC == MR0
#define DELAY_MS 2
#define PRESCALE 2625	// TC inkrementowane co (2715 - czas_przetwarzania) cykli zegara
						// wartoœæ dla próbkowania 11050 Hz

#define PLLFEEDCODE01 0xAA
#define PLLFEEDCODE02 0x55

void myTimerExec(void);
void StopInterrupts(void);

#endif /* MYTIMER_H_ */
