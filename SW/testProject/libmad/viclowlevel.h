/* Enable and disable functions "ripped" from a sample by R O Software.
* Copyright 2004, R O SoftWare
* No guarantees, warrantees, or promises, implied or otherwise.
* May be used for hobby or commercial purposes provided copyright
* notice remains intact. */
# ifndef VICLOWLEVEL_H
# define VICLOWLEVEL_H

extern unsigned enableIRQ(void);
extern unsigned disableIRQ(void);
extern unsigned restoreIRQ(unsigned oldCPSR);

#endif
