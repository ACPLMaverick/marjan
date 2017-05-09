	.inesprg 1   ; 1x 16KB bank of PRG code
	.ineschr 1   ; 1x 8KB bank of CHR data
	.inesmap 0   ; mapper 0 = NROM, no bank swapping
	.inesmir 1   ; background mirroring (ignore for now)

	;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

	;;;;;;;;;;;;;;;;;;;;;
	;;DECLARE VARIABLES;;
	;;;;;;;;;;;;;;;;;;;;;
	.rsset $0000 ; start variables at RAM location 0
pointerLo  .rs 1   ; pointer variables are declared in RAM
pointerHi  .rs 1   ; low byte first, high byte immediately after

buttons1   .rs 1  ; player 1 gamepad buttons, one bit per button
buttons2   .rs 1  ; player 2 gamepad buttons, one bit per button
paddle1ytop   .rs 1  ; player 1 paddle top vertical position
paddle2ytop   .rs 1  ; player 2 paddle top vertical position

paddle1canMoveUp .rs 1 ; player 1 can move up
paddle1canMoveDown .rs 1 ; player 1 can move down
paddle2canMoveUp .rs 1 ; player 2 can move up
paddle2canMoveDown .rs 1 ; player 2 can move down

	;;;;;;;;;;;;;;;;;;;;;
	;;DECLARE CONSTANTS;;
	;;;;;;;;;;;;;;;;;;;;;

RIGHTWALL      = $F4  ; when ball reaches one of these, do something
TOPWALL        = $05
BOTTOMWALL     = $D0
LEFTWALL       = $04

PADDLE1X       = $08  ; horizontal position for paddles, doesn't move
PADDLE2X       = $F0

	;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

	.bank 0
	.org $C000	 ; put the code starting at memory location $C000

	;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
	;;INITIAL SETTINGS AND LOADING GRAPHICS;;
	;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

vblankwait:		 ; subroutine for vblank waiting
	BIT $2002
	BPL vblankwait
	RTS

RESET:
	SEI			 ; disable IRQ
	CLD			 ; disable decimal mode

	LDX #$40
	STX $4017    ; disable APU frame IRQ
	LDX #$FF
	TXS          ; Set up stack
	INX          ; now X = 0
	STX $2000    ; disable NMI
	STX $2001    ; disable rendering
	STX $4010    ; disable DMC IRQs

	JSR vblankwait

clrmem:
	LDA #$00
	STA $0000, x
	STA $0100, x
	STA $0300, x
	STA $0400, x
	STA $0500, x
	STA $0600, x
	STA $0700, x
	LDA #$FE
	STA $0200, x
	INX
	BNE clrmem
   
	JSR vblankwait


LoadPalettes:
	LDA $2002	 ; read PPU status to reset the high/low latch to high - resets the PPU to expect the high byte first.
	LDA #$3F	 ; set high bytes
	STA $2006
	LDA #$10	 ; set low bytes
	STA $2006
	LDX #$00	 ; set iterator to 0
LoadPalettesLoop:
	LDA palette, x	 ; load data from address (PaletteData + the value in x)
	STA $2007		 ; write to PPU
	INX
	CPX #$20		 ; compare X to hex $20, decimal 32
	BNE LoadPalettesLoop

LoadSprites:
	LDX #$00
LoadSpritesLoop:
	LDA sprites, x	 ; load data from address (sprites + x)
	STA $0200, x	 ; store data into address ($0200 + x)
	INX
	CPX #$10
	BNE LoadSpritesLoop

	;;;Set some initial paddles stats
	LDA #$72
	STA paddle1ytop
	STA paddle2ytop
	LDA #$01
	STA paddle1canMoveUp
	STA paddle1canMoveDown
	STA paddle2canMoveUp
	STA paddle2canMoveDown

LoadBackground:
	LDA $2002
	LDA #$20
	STA $2006
	LDA #$00
	STA $2006

	LDA #$00
	STA pointerLo		; put the low byte of the address of background into pointer
	LDA #$E0
	STA pointerHi		; put the high byte of the address into pointer

	LDX #$00
LoadBackgroundOutsideLoop:
	LDY #$00			; reset Y to get background from the same address range in every iteration

LoadBackgroundInsideLoop:
	LDA [pointerLo], y
	STA $2007

	INY
	CPY #$20			; 32 bytes of background
	BNE LoadBackgroundInsideLoop

	INX
	CPX #$20			; 32 x 32 to set whole screen
	BNE LoadBackgroundOutsideLoop

LoadAttribute:
	LDA $2002
	LDA #$23
	STA $2006             ; write the high byte of $23C0 address
	LDA #$C0
	STA $2006             ; write the low byte of $23C0 address
	LDX #$00              ; start out at 0
LoadAttributeLoop:
	LDA attribute, x
	STA $2007
	INX
	CPX #$08              ; Compare X to hex $08, decimal 8 - copying 8 bytes
	BNE LoadAttributeLoop

	LDA #%10010000	 ; enable NMI, sprites from Pattern Table 0
	STA $2000

	LDA #%00011110   ; no intensify (black background), enable sprites
	STA $2001
	
	;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
	;;ENDLESS LOOP AND GRAPHICS UPDATE;;
	;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

Forever:
	JMP Forever		 ; loop -> wait for NMI

NMI:			; graphics update
	LDA #$00
	STA $2003	  
	LDA #$02
	STA $4014    ; start DMA transfer (copy block of RAM from CPU to PPU) - whole 64 sprites (256 bytes)

	;;This is the PPU clean up section, so rendering the next frame starts properly.
	LDA #%10011000   ; enable NMI, sprites from Pattern Table 1, background from Pattern Table 1
	STA $2000
	LDA #%00011110   ; enable sprites, enable background, no clipping on left side
	STA $2001
	LDA #$00
	STA $2005
	STA $2005	 ; we are not doing any scrolling at the end of NMI

	;;;;;;;;;;;;;;;;;;;;
	;;READ CONTROLLERS;;
	;;;;;;;;;;;;;;;;;;;;

	JSR ReadController1
	JSR ReadController2

	;;;;;;;;;;;;;;;;;;;;;;;;;;;;
	;;GAME ENGINE STATE CHECKS;;
	;;;;;;;;;;;;;;;;;;;;;;;;;;;;

GameEngine:
	JSR EnginePlaying

GameEngineDone:
	JSR UpdateSprites

	RTI			 ; return from NMI interrupt

	;;;;;;;;;;;;;;;;;;;
	;;MOVEMENT UPDATE;;
	;;;;;;;;;;;;;;;;;;;
EnginePlaying:

MovePaddleOneUp:
	LDA buttons1
	AND #%00001000			; check if Up Button is pressed
	BEQ MovePaddleOneUpDone ; Branch to done if button is NOT pressed

	LDA #$01
	STA paddle1canMoveDown

	LDA paddle1canMoveUp
	CMP #$01				; check if paddle can move up
	BCC MovePaddleOneUpDone

	LDA paddle1ytop
	SEC
	SBC #$08
	CMP #TOPWALL
	STA paddle1ytop
	BCS MovePaddleOneUpDone ; if paddle y > top wall, still on screen, skip next section
	LDA #$00
	STA paddle1canMoveUp
	
MovePaddleOneUpDone:

MovePaddleOneDown:
	LDA buttons1
	AND #%00000100			; check if Up Button is pressed
	BEQ MovePaddleOneDownDone ; Branch to done if button is NOT pressed

	LDA #$01
	STA paddle1canMoveUp

	LDA paddle1canMoveDown
	CMP #$01				; check if paddle can move up
	BCC MovePaddleOneDownDone

	LDA paddle1ytop
	CLC
	ADC #$08
	CMP #BOTTOMWALL
	STA paddle1ytop
	BCC MovePaddleOneDownDone ; if paddle y < bottom wall, still on screen, skip next section
	LDA #$00
	STA paddle1canMoveDown

MovePaddleOneDownDone:

MovePaddleTwoUp:
	LDA buttons2
	AND #%00001000			; check if Up Button is pressed
	BEQ MovePaddleTwoUpDone ; Branch to done if button is NOT pressed

	LDA #$01
	STA paddle2canMoveDown

	LDA paddle2canMoveUp
	CMP #$01				; check if paddle can move up
	BCC MovePaddleTwoUpDone

	LDA paddle2ytop
	SEC
	SBC #$08
	CMP #TOPWALL
	STA paddle2ytop
	BCS MovePaddleTwoUpDone ; if paddle y > top wall, still on screen, skip next section
	LDA #$00
	STA paddle2canMoveUp

MovePaddleTwoUpDone:

MovePaddleTwoDown:
	LDA buttons2
	AND #%00000100			; check if Up Button is pressed
	BEQ MovePaddleTwoDownDone ; Branch to done if button is NOT pressed

	LDA #$01
	STA paddle2canMoveUp

	LDA paddle2canMoveDown
	CMP #$01				; check if paddle can move up
	BCC MovePaddleTwoDownDone

	LDA paddle2ytop
	CLC
	ADC #$08
	CMP #BOTTOMWALL
	STA paddle2ytop
	BCC MovePaddleTwoDownDone ; if paddle y < bottom wall, still on screen, skip next section
	LDA #$00
	STA paddle2canMoveDown

MovePaddleTwoDownDone:

	JMP GameEngineDone

	;;;;;;;;;;;;;;;;;;
	;;SPRITES UPDATE;;
	;;;;;;;;;;;;;;;;;;
UpdateSprites:
	LDA paddle1ytop ;draw paddle 1 Top
	STA $0210
	LDA #$46
	STA $0211
	LDA #$00
	STA $0212
	LDA #PADDLE1X
	STA $0213

	LDA paddle1ytop
	CLC
	ADC #$08	;draw paddle 1 center
	STA $0214
	LDA #$46
	STA $0215
	LDA #$00
	STA $0216
	LDA #PADDLE1X
	STA $0217

	LDA paddle1ytop	;draw paddle 1 bottom
	CLC 
	ADC #$10
	STA $0218
	LDA #$46
	STA $0219
	LDA #$00
	STA $021A
	LDA #PADDLE1X
	STA $021B

	LDA paddle2ytop ;draw paddle 2 Top
	STA $021C
	LDA #$4A
	STA $021D
	LDA #$00
	STA $021E
	LDA #PADDLE2X
	STA $021F

	LDA paddle2ytop
	CLC
	ADC #$08	;draw paddle 2 center
	STA $0220
	LDA #$4A
	STA $0221
	LDA #$00
	STA $0222
	LDA #PADDLE2X
	STA $0223

	LDA paddle2ytop	;draw paddle 2 bottom
	CLC 
	ADC #$10
	STA $0224
	LDA #$4A
	STA $0225
	LDA #$00
	STA $0226
	LDA #PADDLE2X
	STA $0227

	RTS

	;;;;;;;;;;;;;
	;;SET SCORE;;
	;;;;;;;;;;;;;

	;;;;;;;;;;;;;;;;;;;;;;;;;;
	;;CONTROLLER SUBROUTINES;;
	;;;;;;;;;;;;;;;;;;;;;;;;;;
ReadController1:
	LDA #$01
	STA $4016
	LDA #$00
	STA $4016		; latch the current buttons position
	LDX #$08
ReadController1Loop:
	LDA $4016
	LSR A			; right shift to move A 0 bit to carry bit
	ROL buttons1	; left shift to move carry bit to buttons1 0 bit
	DEX
	BNE ReadController1Loop
	RTS

ReadController2:
	LDA #$01
	STA $4016
	LDA #$00
	STA $4016		; latch the current buttons position
	LDX #$08
ReadController2Loop:
	LDA $4017
	LSR A
	ROL buttons2
	DEX
	BNE ReadController2Loop
	RTS

	;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;




	.bank 1
	.org $E000

nametables: ;[32]
	.db $24,$24,$24,$24,$24,$24,$24,$24,$24,$24,$24,$24,$24,$24,$24,$24  ;;row 1
	.db $24,$24,$24,$24,$24,$24,$24,$24,$24,$24,$24,$24,$24,$24,$24,$24  ;;all sky ($24 = sky)
	
palette: ;[32]
	.db $22,$29,$1A,$0F,$22,$02,$38,$3C,$22,$1C,$15,$14,$22,$02,$38,$3C		; sprite palette data
	.db $22,$29,$1A,$0F,$22,$36,$17,$0F,$22,$30,$21,$0F,$22,$27,$17,$0F		; background palette data

sprites: ;[16]
	;vertical, tile, attributes, horizontal
	.db $80, $A5, $00, $80	;sprite 0
	.db $80, $A6, $00, $88	;sprite 1
	.db $88, $A7, $00, $80	;sprite 2
	.db $88, $A8, $00, $88	;sprite 3

attribute:
	.db %00000000

	.org $FFFA	 ; first of the three vectors (interruptions) starts here
	.dw NMI		 ; when an NMI (starting VBlank time so its time for updating graphics) happens (once per frame if enabled) the 
                   ; processor will jump to the label NMI
	.dw RESET	 ; when the processor first turns on or is reset, it will jump
                   ; to the label RESET
	.dw 0		 ; external interrupt IRQ (but it's not used here)
	; more code here




	;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;



	
	.bank 2		 ; graphics here
	.org $0000
	.incbin "mario.chr" ; includes 8KB graphics file from SMB1