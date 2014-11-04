/*****************************************************************************/
#include "lpc2xxx.h"
#include "hwinterface.h"
#include "sd.h"
#include "config.h"
/*****************************************************************************/

//#ifndef HW_ENDPOINT_LPC2000_SPINUM
//#error "HW_ENDPOINT_LPC2000_SPINUM has to be defined in config.h"
//#endif

#if ( HW_ENDPOINT_LPC2000_SPINUM == 0 )
// LPC213x ### SPI0 ###

// SP0SPCR  Bit-Definitions
#define CPHA    3
#define CPOL    4
#define MSTR    5
// SP0SPSR  Bit-Definitions
#define SPIF	7

#define SPI_IODIR      IODIR0
#define SPI_SCK_PIN    4   /* Clock       P0.4  out */
#define SPI_MISO_PIN   5   /* from Card   P0.5  in  */
#define SPI_MOSI_PIN   6   /* to Card     P0.6  out */
#define SPI_SS_PIN	   11   /* Card-Select P0.7 - GPIO out */

#define SPI_PINSEL     PINSEL0
#define SPI_SCK_FUNCBIT   8
#define SPI_MISO_FUNCBIT  10
#define SPI_MOSI_FUNCBIT  12
#define SPI_SS_FUNCBIT    14

#define SPI_PRESCALE_REG  S0SPCCR
#define SPI_PRESCALE_MIN  8

#define SELECT_CARD()   IOCLR0 = (1<<SPI_SS_PIN)
#define UNSELECT_CARD()	IOSET0 = (1<<SPI_SS_PIN)

#elif ( HW_ENDPOINT_LPC2000_SPINUM == 1 )
// LPC213x ### SSP ### ("SPI1")

// SSPCR0  Bit-Definitions
#define CPOL    6
#define CPHA    7
// SSPCR1  Bit-Defintions
#define SSE     1
#define MS      2
#define SCR     8
// SSPSR  Bit-Definitions
#define TNF     1
#define RNE     2
#define BSY		4

#define SPI_IODIR      IODIR0
#define SPI_SCK_PIN    17   /* Clock       P0.17  out */
#define SPI_MISO_PIN   18   /* from Card   P0.18  in  */
#define SPI_MOSI_PIN   19   /* to Card     P0.19  out */
/* Card-Select P0.20 - GPIO out during startup
   Function 03 during normal operation */
#define SPI_SS_PIN	   20   

#define SPI_PINSEL     PINSEL1
#define SPI_SCK_FUNCBIT   2
#define SPI_MISO_FUNCBIT  4
#define SPI_MOSI_FUNCBIT  6
#define SPI_SS_FUNCBIT    8

#define SPI_PRESCALE_REG  SSPCPSR
/// TODO: too fast on prototyp wires #define SPI_PRESCALE_MIN  2
#define SPI_PRESCALE_MIN  4

/* only needed during init: */
#define SELECT_CARD()   IOCLR0 = (1<<SPI_SS_PIN)
#define UNSELECT_CARD()	IOSET0 = (1<<SPI_SS_PIN)


#else
#error "Invalid Interface-Number"
#endif




signed char if_initInterface(hwInterface* file, char* opts)
{
	unsigned long sc;
	
	if_spiInit(file); /* init at low speed */
	signed char returnValue = sd_Init(file);
	
	if(returnValue<0)	{
		DBG((TXT("Card failed to init, breaking up...\n")));
		return returnValue;
	}
	if(sd_State(file)<0){
		DBG((TXT("Card didn't return the ready state, breaking up...\n")));
		return(-2);
	}
	
	sd_getDriveSize(file, &sc);
	file->sectorCount = sc/512;
	if( (sc%512) != 0) {
		file->sectorCount--;
	}
	DBG((TXT("Drive Size is %lu Bytes (%lu Sectors)\n"), sc, file->sectorCount));
	
	 /* increase speed after init */
#if ( HW_ENDPOINT_LPC2000_SPINUM == 1 )
	SSPCR0 = ((8-1)<<0) | (0<<CPOL);
#endif
	if_spiSetSpeed(SPI_PRESCALE_MIN);
	// if_spiSetSpeed(100); /* debug - slower */
	
	DBG((TXT("Init done...\n")));
	return(0);
}
/*****************************************************************************/ 

signed char if_readBuf(hwInterface* file,unsigned long address,unsigned char* buf)
{
	return(sd_readSector(file,address,buf,512));
}
/*****************************************************************************/

signed char if_writeBuf(hwInterface* file,unsigned long address,unsigned char* buf)
{
	return(sd_writeSector(file,address, buf));
}
/*****************************************************************************/ 

signed char if_setPos(hwInterface* file,unsigned long address)
{
	return(0);
}
/*****************************************************************************/ 

// Utility-functions which does not toogle CS.
// Only needed during card-init. During init
// the automatic chip-select is disabled for SSP

static unsigned char my_if_spiSend(hwInterface *iface, unsigned char outgoing)
{
	unsigned char incoming;

	// SELECT_CARD(); // not here!
	
#if ( HW_ENDPOINT_LPC2000_SPINUM == 0 )
	S0SPDR = outgoing;
	while( !(S0SPSR & (1<<SPIF)) ) ;
	incoming = S0SPDR;
#endif
#if ( HW_ENDPOINT_LPC2000_SPINUM == 1 )
	while( !(SSPSR & (1<<TNF)) ) ;
	SSPDR = outgoing;
	while( !(SSPSR & (1<<RNE)) ) ;
	incoming = SSPDR;
#endif

	// UNSELECT_CARD(); // not here!

	return(incoming);
}
/*****************************************************************************/ 

void if_spiInit(hwInterface *iface)
{
	unsigned char i; 

	// setup GPIO
	SPI_IODIR |= (1<<SPI_SCK_PIN)|(1<<SPI_MOSI_PIN)|(1<<SPI_SS_PIN);
	SPI_IODIR &= ~(1<<SPI_MISO_PIN);
	
	// set Chip-Select high - unselect card
	UNSELECT_CARD();

	// reset Pin-Functions	
	SPI_PINSEL &= ~( (3<<SPI_SCK_FUNCBIT) | (3<<SPI_MISO_FUNCBIT) |
		(3<<SPI_MOSI_FUNCBIT) | (3<<SPI_SS_FUNCBIT) );

#if ( HW_ENDPOINT_LPC2000_SPINUM == 0 )
	DBG((TXT("spiInit for SPI(0)\n")));
	SPI_PINSEL |= ( (1<<SPI_SCK_FUNCBIT) | (1<<SPI_MISO_FUNCBIT) |
		(1<<SPI_MOSI_FUNCBIT) );
	// enable SPI-Master
	S0SPCR = (1<<MSTR)|(0<<CPOL); // TODO: check CPOL
#endif

#if ( HW_ENDPOINT_LPC2000_SPINUM == 1 )
	DBG((TXT("spiInit for SSP/SPI1\n")));
	// setup Pin-Functions - keep automatic CS disabled during init
	SPI_PINSEL |= ( (2<<SPI_SCK_FUNCBIT) | (2<<SPI_MISO_FUNCBIT) |
		(2<<SPI_MOSI_FUNCBIT) | (0<<SPI_SS_FUNCBIT) );
	// enable SPI-Master - slowest speed
	SSPCR0 = ((8-1)<<0) | (0<<CPOL) | (0x20<<SCR); //  (0xff<<SCR);
	SSPCR1 = (1<<SSE);
#endif
	
	// low speed during init
	if_spiSetSpeed(254); 

	/* Send 20 spi commands with card not selected */
	for(i=0;i<21;i++)
		my_if_spiSend(iface,0xff);

#if ( HW_ENDPOINT_LPC2000_SPINUM == 0 )
	// SPI0 does not offer automatic CS for slaves on LPC2138 
	// ( the SSEL-Pin is input-only )
	// SELECT_CARD();
#endif

#if ( HW_ENDPOINT_LPC2000_SPINUM == 1 )
	// enable automatic slave CS for SSP
	SSPCR1 &= ~(1<<SSE); // disable interface
	SPI_PINSEL |= ( (2<<SPI_SCK_FUNCBIT) | (2<<SPI_MISO_FUNCBIT) |
		(2<<SPI_MOSI_FUNCBIT) | (2<<SPI_SS_FUNCBIT) );
	SSPCR1 |= (1<<SSE); // enable interface
#endif
	
}
/*****************************************************************************/

void if_spiSetSpeed(euint8 speed)
{
	speed &= 0xFE;
	if ( speed < SPI_PRESCALE_MIN  ) speed = SPI_PRESCALE_MIN ;
	SPI_PRESCALE_REG = speed;
}

/*****************************************************************************/

unsigned char if_spiSend(hwInterface *iface, unsigned char outgoing)
{
	unsigned char incoming;

#if ( HW_ENDPOINT_LPC2000_SPINUM == 0 )
	SELECT_CARD();
	S0SPDR = outgoing;
	while( !(S0SPSR & (1<<SPIF)) ) ;
	incoming = S0SPDR;
	UNSELECT_CARD();
#endif

#if ( HW_ENDPOINT_LPC2000_SPINUM == 1 )
	// SELECT_CARD();  // done by hardware
	while( !(SSPSR & (1<<TNF)) ) ;
	SSPDR = outgoing;
	while( !(SSPSR & (1<<RNE)) ) ;
	incoming = SSPDR;
	// UNSELECT_CARD();  // done by hardware
#endif

	

	return(incoming);
}
/*****************************************************************************/

