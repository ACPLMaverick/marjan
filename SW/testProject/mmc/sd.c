/*****************************************************************************/
#include "sd.h"
/*****************************************************************************/

signed char sd_Init(hwInterface *iface)
{
	signed short i;
	unsigned char resp;
	
	/* Try to send reset command up to 100 times */
	i=100;
	do{
		sd_Command(iface,0, 0, 0);
		resp=sd_Resp8b(iface);
	}
	while(resp!=1 && i--);
	
	if(resp!=1){
		if(resp==0xff){
			return(-1);
		}
		else{
			return sd_Resp8bError(iface,resp);
			//return(-2);
		}
	}

	/* Wait till card is ready initialising (returns 0 on CMD1) */
	/* Try up to 32000 times. */
	i=32000;
	do{
		sd_Command(iface,1, 0, 0);
		
		resp=sd_Resp8b(iface);
		if(resp!=0)
			sd_Resp8bError(iface,resp);
	}
	while(resp==1 && i--);
	
	if(resp!=0){
		sd_Resp8bError(iface,resp);
		return(-3);
	}
	
	return(0);
}
/*****************************************************************************/

void sd_Command(hwInterface *iface,unsigned char cmd, unsigned short paramx, unsigned short paramy)
{
	if_spiSend(iface,0xff);

	if_spiSend(iface,0x40 | cmd);
	if_spiSend(iface,(euint8) (paramx >> 8)); /* MSB of parameter x */
	if_spiSend(iface,(euint8) (paramx)); /* LSB of parameter x */
	if_spiSend(iface,(euint8) (paramy >> 8)); /* MSB of parameter y */
	if_spiSend(iface,(euint8) (paramy)); /* LSB of parameter y */

	if_spiSend(iface,0x95); /* Checksum (should be only valid for first command (0) */

	if_spiSend(iface,0xff); /* eat empty command - response */
}
/*****************************************************************************/

unsigned char sd_Resp8b(hwInterface *iface)
{
	unsigned char i;
	unsigned char resp;
	
	/* Respone will come after 1 - 8 pings */
	for(i=0;i<8;i++){
		resp = if_spiSend(iface,0xff);
		if(resp != 0xff)
			return(resp);
	}
		
	return(resp);
}
/*****************************************************************************/

unsigned short sd_Resp16b(hwInterface *iface)
{
	unsigned short resp;
	
	resp = ( sd_Resp8b(iface) << 8 ) & 0xff00;
	resp |= if_spiSend(iface,0xff);
	
	return(resp);
}
/*****************************************************************************/

signed short sd_Resp8bError(hwInterface *iface,unsigned char value)
{
	switch(value)
	{
		case 0x40:
			DBG((TXT("Argument out of bounds.\n")));
			return -10;
			break;
		case 0x20:
			DBG((TXT("Address out of bounds.\n")));
			return -11;
			break;
		case 0x10:
			DBG((TXT("Error during erase sequence.\n")));
			return -12;
			break;
		case 0x08:
			DBG((TXT("CRC failed.\n")));
			return -13;
			break;
		case 0x04:
			DBG((TXT("Illegal command.\n")));
			return -14;
			break;
		case 0x02:
			DBG((TXT("Erase reset (see SanDisk docs p5-13).\n")));
			return -15;
			break;
		case 0x01:
			DBG((TXT("Card is initialising.\n")));
			return -16;
			break;
		default:
			DBG((TXT("Unknown error 0x%x (see SanDisk docs p5-13).\n"),value));
			return -17;
			break;
	}
}
/*****************************************************************************/

signed char sd_State(hwInterface *iface)
{
	short value;
	
	sd_Command(iface,13, 0, 0);
	value=sd_Resp16b(iface);

	switch(value)
	{
		case 0x000:
			return(1);
			break;
		case 0x0001:
			DBG((TXT("Card is Locked.\n")));
			break;
		case 0x0002:
			DBG((TXT("WP Erase Skip, Lock/Unlock Cmd Failed.\n")));
			break;
		case 0x0004:
			DBG((TXT("General / Unknown error -- card broken?.\n")));
			break;
		case 0x0008:
			DBG((TXT("Internal card controller error.\n")));
			break;
		case 0x0010:
			DBG((TXT("Card internal ECC was applied, but failed to correct the data.\n")));
			break;
		case 0x0020:
			DBG((TXT("Write protect violation.\n")));
			break;
		case 0x0040:
			DBG((TXT("An invalid selection, sectors for erase.\n")));
			break;
		case 0x0080:
			DBG((TXT("Out of Range, CSD_Overwrite.\n")));
			break;
		default:
			if(value>0x00FF)
				sd_Resp8bError(iface,(euint8) (value>>8));
			else
				DBG((TXT("Unknown error: 0x%x (see SanDisk docs p5-14).\n"),value));
			break;
	}
	return(-1);
}
/*****************************************************************************/

/* ****************************************************************************
 * WAIT ?? -- FIXME
 * CMDWRITE
 * WAIT
 * CARD RESP
 * WAIT
 * DATA BLOCK OUT
 *      START BLOCK
 *      DATA
 *      CHKS (2B)
 * BUSY...
 */

signed char sd_writeSector(hwInterface *iface,unsigned long address, unsigned char* buf)
{
	unsigned long place;
	unsigned short i;
	unsigned short t=0;
	
	/*DBG((TXT("Trying to write %u to sector %u.\n"),(void *)&buf,address));*/
	place=512*address;
	sd_Command(iface,CMDWRITE, (euint16) (place >> 16), (euint16) place);

	sd_Resp8b(iface); /* Card response */

	if_spiSend(iface,0xfe); /* Start block */
	for(i=0;i<512;i++) 
		if_spiSend(iface,buf[i]); /* Send data */
	if_spiSend(iface,0xff); /* Checksum part 1 */
	if_spiSend(iface,0xff); /* Checksum part 2 */

	if_spiSend(iface,0xff);

	while(if_spiSend(iface,0xff)!=0xff){
		t++;
		/* Removed NOP */
	}
	/*DBG((TXT("Nopp'ed %u times.\n"),t));*/

	return(0);
}
/*****************************************************************************/

/* ****************************************************************************
 * WAIT ?? -- FIXME
 * CMDCMD
 * WAIT
 * CARD RESP
 * WAIT
 * DATA BLOCK IN
 * 		START BLOCK
 * 		DATA
 * 		CHKS (2B)
 */

signed char sd_readSector(hwInterface *iface,unsigned long address, unsigned char* buf, unsigned short len)
{
	unsigned char cardresp;
	unsigned char firstblock;
	unsigned char c;
	unsigned short fb_timeout = 0xffff;
	unsigned long i;
	unsigned long place;

	/*DBG((TXT("sd_readSector::Trying to read sector %u and store it at %p.\n"),address,&buf[0]));*/
	place=512*address;
	sd_Command(iface,CMDREAD, (euint16) (place >> 16), (euint16) place);
	
	cardresp=sd_Resp8b(iface); /* Card response */ 

	/* Wait for startblock */
	do
		firstblock=sd_Resp8b(iface); 
	while(firstblock==0xff && fb_timeout--);

	if(cardresp!=0x00 || firstblock!=0xfe){
		sd_Resp8bError(iface,firstblock);
		return(-1);
	}
	
	for(i=0;i<512;i++){
		c = if_spiSend(iface,0xff);
		if(i<len)
			buf[i] = c;
	}

	/* Checksum (2 byte) - ignore for now */
	if_spiSend(iface,0xff);
	if_spiSend(iface,0xff);

	return(0);
}
/*****************************************************************************/

/* ****************************************************************************
 calculates size of card from CSD 
 (extension by Martin Thomas, inspired by code from Holger Klabunde)
 */
signed char sd_getDriveSize(hwInterface *iface, unsigned long* drive_size )
{
	unsigned char cardresp, i, by;
	unsigned char iob[16];
	unsigned short c_size, c_size_mult, read_bl_len;
	
	sd_Command(iface, CMDREADCSD, 0, 0);
	
	do {
		cardresp = sd_Resp8b(iface);
	} while ( cardresp != 0xFE );

	DBG((TXT("CSD:")));
	for( i=0; i<16; i++) {
		iob[i] = sd_Resp8b(iface);
		DBG((TXT(" %02x"), iob[i]));
	}
	DBG((TXT("\n")));

	if_spiSend(iface,0xff);
	if_spiSend(iface,0xff);
	
	c_size = iob[6] & 0x03; // bits 1..0
	c_size <<= 10;
	c_size += (euint16)iob[7]<<2;
	c_size += iob[8]>>6;

	by= iob[5] & 0x0F;
	read_bl_len = 1;
	read_bl_len <<= by;

	by=iob[9] & 0x03;
	by <<= 1;
	by += iob[10] >> 7;
	
	c_size_mult = 1;
	c_size_mult <<= (2+by);
	
	*drive_size = (euint32)(c_size+1) * (euint32)c_size_mult * (euint32)read_bl_len;
	
	return 0;
}
