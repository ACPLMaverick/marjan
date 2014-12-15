/***********************************************************************/
/*  This file is part of the uVision/ARM development tools             */
/*  Copyright KEIL ELEKTRONIK GmbH 2002-2004                           */
/***********************************************************************/
/*                                                                     */
/*  SYSCALLS.C:  System Calls Remapping                                */
/*                                                                     */
/***********************************************************************/

#include <sys/types.h>


void _exit (int n) {
label:  goto label; /* endless loop */
}


#define HEAP_LIMIT 0x80200000

caddr_t sbrk (int incr) {
	  extern char end;
	  /* Defined by the linker. */
	  static char * heap_end;
	  char * prev_heap_end;

	  if ( heap_end == 0)
	  {
	    heap_end = & end;
	  }
	  prev_heap_end = heap_end;

	  heap_end += incr;

	  return 0;
}
