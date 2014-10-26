/*---------------------------------------------------------
*	Name: MMCMAIN.C
*	Purpose: SD/MMC Access
*-------------------------------------------------------*/
#include <lpc2xxx.h>

#include "type.h"		//póki co nie ma
#include "spc_mmc.h"
#include "mmcmain.h"

extern BYTE MMCWRData[MMC_DATA_SIZE];
extern BYTE MMCRDData[MMC_DATA_SIZE];

int main(void)
{
	DWORD i, BlockNum = 0;
	
	PINSEL1 = 0x4004000;
	IODIR1 = LED_MSK;		/* LED's defined as Outputs */ 

	SPI_Init();				/* initialize SPI for MMC card */ 
	IOSET1 = LED_CFG;
	if (mmc_init() != 0)
	{
		IOSET0 = SPI_SEL;	/*set SSEL to high*/
		while (1);
	}

	/* write, read back, and compare the complete 64KB on the MMC
	* card each block is 512 bytes, the total is 512 * 128 */
	for (BlockNum = 0; BlockNum < MAX_BLOCK_NUM; BlockNum++)
	{
		IOCLR1 = LED_MSK;
		IOSET1 = LED_WR;
		if (mmc_write_block(BlockNum) == 0)
		{
			IOCLR1 = LED_MSK;
			IOSET1 = LED_RD;
			mmc_read_block(BlockNum);
		}
		else
		{
			IOSET0 = SPI_SEL;
			while (1);
		}

		for (i = 0; i < MMC_DATA_SIZE; i++)	/* Validate */ 
		{
			if (MMCRDData[i] != MMCWRData[i])
			{
				IOSET0 = SPI_SEL;
				while (1);
			}
		}

		for (i = 0; i < MMC_DATA_SIZE; i++)	/* clear read buffer */ 
		{
			MMCRDData[i] = 0x00;
		}

		IOCLR1 = LED_MSK;
		IOSET1 = LED_DONE;
		while (1);						/* Loop forever */ 
	}
}