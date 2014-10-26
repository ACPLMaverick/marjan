/******************************************************************************
 *
 * Copyright:
 *    (C) 2000 - 2005 Embedded Artists AB
 *
 * Description:
 *
 * ESIC:
 *    oshal_arm7_gcc
 *
 * Version:
 *    1.1.0
 *
 * Generate date:
 *    2005-03-15 at 20:27:15
 *
 * NOTE:
 *    DO NOT EDIT THIS FILE. IT IS AUTO GENERATED.
 *    CHANGES TO THIS FILE WILL BE LOST IF THE FILE IS RE-GENERATED
 *
 * Signature:
 *   7072655F656D70746976655F6F73,312E342E302E30,020235
 *   ,35,10104021013134373435363030,07323838,3732,01013
 *   830,0101013138303030303030,3135,33,3135,0232323530
 *   ,01020130,0231343734353539,3238313831,020101100302
 *   103030310010133,0163130,3230,3330,3430,3530,3630,3
 *   730,3830,3930,313030,313130,313230,313330,313430,3
 *   13530,313630,,35,35,35,35,35,35,35,35,35,35,35,35,
 *   35,35,35,35,,,,1001001100011000000000]484152445741
 *   5245,4C5043323130365F32,545538,756E7369676E6564206
 *   3686172,414C49474E4D454E54,34,54424F4F4C,756E73696
 *   76E65642063686172,54553332,756E7369676E656420696E7
 *   4,544D505F46494C4553,2A2E656C663B2A2E6C73743B2A2E6
 *   D61703B2A2E6F3B2A2E6F626A3B2A2E64,454E4449414E,4C4
 *   954544C45,54533332,7369676E656420696E74,545338,736
 *   9676E65642063686172,54553136,756E7369676E656420736
 *   86F7274,54533136,7369676E65642073686F7274,44455343
 *   52495054494F4E,,44454255475F4C4556454C,30,434F4445
 *   5F524F4F54,,47454E5F52554C4553,,4C494E455F5445524D
 *   ,43524C46,4252414345,,43524541544F52,416E646572732
 *   0526F7376616C6C,4352454154494F4E5F44415445,3230303
 *   52D30332D31352032303A31373A3432,524F4F54,433A2F446
 *   F63756D656E747320616E642053657474696E67732F416E646
 *   5727320526F7376616C6C2F4D696E6120646F6B756D656E742
 *   F456D62656464656420417274697374732F50726F647563747
 *   32F4C50433231303620525332333220517569636B537461727
 *   420426F6172642F72746F732F]505245464958,,4445425547
 *   5F4C4556454C,30,555345525F434F4D4D454E54,]64656661
 *   756C74,
 *
 * Checksum:
 *    242774
 *
 *****************************************************************************/


#ifndef _a7hal_oshal_h_
#define _a7hal_oshal_h_

/******************************************************************************
 * Includes
 *****************************************************************************/

#include "general.h"


/*****************************************************************************
 *
 * Description:
 *    Initializes a stack frame 
 *
 * Params:
 *    [in] task     - Address of process entry point 
 * 
 *               Params (callback):
 *                  arg - 
 *    [in] ptos     - Pointer to top of stack 
 *    [in] pParam   - Parameter to pass to process entry (a void pointer) 
 *    [in] onReturn - Address of return address, if process returns (only 
 *                     allowed for cyclic processes) 
 *
 * Returns:
 *    Stack pointer after initialization (new top-of-stack) 
 *
 ****************************************************************************/
void* stkFrameInit_oshal(void  (*task) (void* arg),
                         void* ptos,
                         void* pParam,
                         void  (*onReturn) (void));


/*****************************************************************************
 *
 * Description:
 *    Disable interrupts 
 *
 * Returns:
 *    The current status register, before disabling interrupts. 
 *
 ****************************************************************************/
tU32 halDisableInterrupts_oshal(void);


/*****************************************************************************
 *
 * Description:
 *    Enable interrupts. 
 *
 ****************************************************************************/
void halEnableInterrupts_oshal(void);


/*****************************************************************************
 *
 * Description:
 *    Restore interrupt state. 
 *
 * Params:
 *    [in] restoreValue - The value of the new status register. 
 *
 ****************************************************************************/
void halRestoreInterrupts_oshal(tU32 restoreValue);


/*****************************************************************************
 *
 * Description:
 *    Setup the timer 
 *
 ****************************************************************************/
void initTimer_oshal(void);


/*****************************************************************************
 *
 * Description:
 *    The routine is called whenever an IRQ occurs, and it should dispatch the 
 *    correct IRQ handler. 
 *
 ****************************************************************************/
void handleIRQs_oshal(void);

/******************************************************************************
 * Description:
 *   This routine is called by osStart to start the first process. It is
 *   assumed that "pNxtToRun" is pointing at the process control block(PCB) 
 *   of the first process to start. "pRunProc" is alterered to point at the 
 *   newly started process' PCB.
 ******************************************************************************/
void startHighProc_oshal(void);

/******************************************************************************
 * Description:
 *   This routine is called from schedule to make a context switch.
 *   "pRunProc" is assumed to point at the currently running process and 
 *   "pNxtToRun" is assumed to point at the PCB of the process to switch to.
 *
 ******************************************************************************/
void ctxSwitch_oshal(void);

/******************************************************************************
 * Description:
 *   This routine is called from osISRExit_oshal to make a context switch.
 *   osISRExit_oshal is called to signal the end of an ISR (interrupt service
 *   routine).
 *   "pRunProc" is assumed to point at the currently running process and 
 *   "pNxtToRun" is assumed to point at the PCB of the process to switch to.
 *
 ******************************************************************************/
void ctxSwitchIsr_oshal(void);

#endif
