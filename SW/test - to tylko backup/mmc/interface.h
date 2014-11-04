#ifndef __TYPES_H__
#define __TYPES_H__

/*****************************************************************************/
#include "types.h"
#include "config.h"
/*****************************************************************************/

#elif defined(HW_ENDPOINT_LPC2000_SD)
	#include "hwinterface.h"
#else
	#error "NO INTERFACE DEFINED - see interface.h"
#endif


