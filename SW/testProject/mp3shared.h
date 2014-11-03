#ifndef _MP3SHARED_H
#define _MP3SHARED_H

// global variables and flags used to communicate between peripherals
typedef struct {
	unsigned char* name;
	unsigned char* author;
	unsigned long time;
	unsigned char ID;
} SongInfo;

extern SongInfo currentSongInfo;
extern unsigned char mmcInitialized;
extern unsigned char changeLeft;
extern unsigned char changeRight;
extern unsigned char rewindForward;
extern unsigned char rewindBackward;
extern unsigned char volumeUp;
extern unsigned char volumeDown;
extern unsigned int isError;
extern char* error;
extern unsigned char currentVolume;

extern void InitializeSharedData();


#endif
