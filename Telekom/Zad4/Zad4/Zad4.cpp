// Zad4.cpp : Defines the entry point for the console application.
//

#include <cstdlib>
#include <iostream>
#include <conio.h>
#include <windows.h>
#include <string>

using namespace std;

HANDLE HandlePort;
bool canRead = true;

char GivePortNumber();
int GiveFuncionality();
bool CreatePort(wchar_t *port, HANDLE &HandlePort);
bool ConfigureConnection(HANDLE HandlePort, int baudRate);
char* PortRead(char* data, int dataSize);
bool PortSend(unsigned char myChar, HANDLE HandlePort);
bool PortSendString(string str, HANDLE HandlePort);
void WaitForOK(HANDLE HandlePort);
DWORD WINAPI reciever(LPVOID lpParam);

int main(int argc, char* argv[])
{
	cout << "MODEM CONNECTION" << endl;
	wchar_t port[5] = { 'c', 'o', 'm' };
	port[3] = GivePortNumber();
	port[4] = NULL;

	int functionality = GiveFuncionality();

	if (!CreatePort(port, HandlePort)) return -1;
	if (!ConfigureConnection(HandlePort, CBR_9600)) return -1;

	HANDLE thread;
	DWORD threadID;
	thread = CreateThread(NULL, 0, reciever, NULL, 0, &threadID);

	if (thread == NULL)
	{
		cout << "Error creating thread" << endl;
		return -1;
	}

	char myChar = NULL;
	if (functionality != 3)
	{
		if (functionality == 1)
		{
			PortSendString("ATM0", HandlePort);
			PortSend(10, HandlePort);
			PortSend(13, HandlePort);
			WaitForOK(HandlePort);
			PortSendString("ATC1", HandlePort);
			PortSend(10, HandlePort);
			PortSend(13, HandlePort);
			WaitForOK(HandlePort);
			PortSendString("ATD55", HandlePort);
			PortSend(10, HandlePort);
			PortSend(13, HandlePort);
		}
		else if (functionality == 2)
		{
			PortSendString("ATM0", HandlePort);
			PortSend(10, HandlePort);
			PortSend(13, HandlePort);
			WaitForOK(HandlePort);
			PortSendString("ATH1", HandlePort);
			PortSend(10, HandlePort);
			PortSend(13, HandlePort);
			WaitForOK(HandlePort);
			PortSendString("ATA", HandlePort);
			PortSend(10, HandlePort);
			PortSend(13, HandlePort);
		}
		else
		{
			cout << "Something went wrong." << endl;
			return(-1);
		}
	}
	/* do wys³ania:
	01 -	ATM0
			ATC1
			ATD55
	02 -	ATH1
			ATA
	*/

	while (true)
	{
		myChar = _getch();
		if (myChar == 27)
		{
			cout << "Connection terminated" << endl;
			CloseHandle(HandlePort);
			break;
		}
		if (myChar == '\r')
		{
			PortSend(10, HandlePort);
			PortSend(13, HandlePort);
			cout << endl;
		}
		cout << myChar;
		PortSend(myChar, HandlePort);
	}

	cout << "Program finished." << endl;
	_getch();
	return 0;
}


char GivePortNumber()
{
	cout << "Give port number: " << endl;
	char myChar;
	do
	{
		myChar = _getch();
	} while (myChar < 49 || myChar > 57);
	return myChar;
}

int GiveFuncionality()
{
	cout << "Give funcionality" << endl
		<< "1: Dialer" << endl
		<< "2: Reciever" << endl
		<< "3: Manual mode" << endl;
	char myChar;
	do
	{
		myChar = _getch();
	} while (myChar != 49 && myChar != 50 && myChar != 51);
	return myChar - 48;
}

bool PortSendString(string str, HANDLE HandlePort)
{
	cout << ">> Sending: " << str << endl;
	unsigned char myChar;
	for (int i = 0; i < str.length(); i++)
	{
		PortSend(str.at(i), HandlePort);
	}
	return true;
}

void WaitForOK(HANDLE HandlePort)
{
	char buffer[3] = { 0, 0, 0 };

	cout << ">> Waiting for OK..." << endl;
	canRead = false;
	while (buffer[0] != 'O' || buffer[1] != 'K')
	{
		PortRead(buffer, 3);
		Sleep(1000);
	}
	canRead = true;
	cout << ">> OK recieved" << endl;
}

bool CreatePort(wchar_t* port, HANDLE &HandlePort)
{
	HandlePort = CreateFile(port, GENERIC_READ | GENERIC_WRITE, 0, NULL, OPEN_EXISTING, 0, NULL);
	if (HandlePort == INVALID_HANDLE_VALUE)
	{
		cout << "Error creating HANDLE" << endl;
		return false;
	}
	else return true;
}

bool ConfigureConnection(HANDLE HandlePort, int baudRate)
{
	DCB SerialPort;
	SerialPort.DCBlength = sizeof(DCB);
	// pobiera domyœlne ustawienie z portu przypisanego do HandlePort
	if (GetCommState(HandlePort, &SerialPort) == 0)
	{
		cout << "GetCommState error" << endl;
		CloseHandle(HandlePort);
		return false;
	}

	SerialPort.BaudRate = baudRate; 
	SerialPort.fBinary = TRUE; // w³¹czenie trybu binarnego
	SerialPort.fParity = TRUE; // w³¹czenie sprawdzania parzystoœci
	SerialPort.fOutxCtsFlow = FALSE; // monitorowanie sygna³u CTS (clear-to-send)
	SerialPort.fOutxDsrFlow = FALSE;  // jw dla DSR (data-set-ready)
	SerialPort.fDtrControl = DTR_CONTROL_ENABLE; // jw dla DTR (data-terminal-ready)
	SerialPort.fDsrSensitivity = false; // wy³¹czenie oddzia³ywania sygna³u DSR na sterownik po³¹czenia
	SerialPort.fTXContinueOnXoff = true; //transmisja jest kontynuowana po tym jak bufor wejœciowy przekroczy limit o XOffLim bajtów i sterownik wyœle XoffChar by zakoñczyæ odbieranie
	SerialPort.fOutX = false; // wy³¹czenie kontroli przep³ywu XON/XOFF podczas wysy³ania
	SerialPort.fInX = false; // jw dla odbierania
	SerialPort.fErrorChar = false; // wy³¹czenie zaznaczania bajtów z b³êdami przez ErrorChary
	SerialPort.fNull = false; // wy³¹czenie odrzucania zerowych bajtów (null bytes)
	SerialPort.fRtsControl = RTS_CONTROL_ENABLE; // w³¹czenie kontroli przep³ywu RTS (request-to-send)
	SerialPort.fAbortOnError = false; // wy³¹czenie zawieszania transmisji przy b³êdzie
	SerialPort.ByteSize = 8;
	SerialPort.Parity = NOPARITY; 
	SerialPort.StopBits = ONESTOPBIT;

	// konfiguruje urz¹dzenie zgodnie z ustawieniami w bloku DCB obiektu SerialPort
	if (SetCommState(HandlePort, &SerialPort) == 0)
	{
		cout << "SetCommState error" << endl;
		CloseHandle(HandlePort);
		return false;
	}

	COMMTIMEOUTS SerialPortTimeouts;

	// taka sama operacja jak wy¿ej, tyle ¿e dla Timeoutów portu

	if (GetCommTimeouts(HandlePort, &SerialPortTimeouts) == 0)
	{
		cout << "GetCommTimeouts error" << endl;
		CloseHandle(HandlePort);
		return false;
	}

	SerialPortTimeouts.ReadIntervalTimeout = MAXDWORD;	// maksymalna iloœæ czasu pomiêdzy kolejnymi bajtami, w milisekundach (du¿o)
	SerialPortTimeouts.ReadTotalTimeoutMultiplier = 0;	// mno¿nik u¿ywany do ustalenia ca³kowitego timeoutu dla odczytu
	SerialPortTimeouts.ReadTotalTimeoutConstant = 0;	// sta³a u¿ywana do j/w
	SerialPortTimeouts.WriteTotalTimeoutMultiplier = 10;	// jw dla zapisu
	SerialPortTimeouts.WriteTotalTimeoutConstant = 100;

	if (SetCommTimeouts(HandlePort, &SerialPortTimeouts) == 0)
	{
		cout << "SetCommTimeouts error" << endl;
		CloseHandle(HandlePort);
		return false;
	}

	cout << "Connection succesfully established :)" << endl;
	return true;
}

char* PortRead(char* data, int dataSize)
{
	DWORD recieved = 0;
	for (int i = 0; i < dataSize; i++) data[i] = 0;
	ReadFile(HandlePort, data, dataSize - 1, &recieved, NULL);
	return data;
}

bool PortSend(unsigned char myChar, HANDLE HandlePort)
{
	DWORD bytesWritten = 0;
	while (bytesWritten == 0) WriteFile(HandlePort, &myChar, 1, &bytesWritten, NULL);
	return true;
}

DWORD WINAPI reciever(LPVOID lpParam)
{
	char buffer[32];
	while (true)
	{
		if (canRead)
		{
			PortRead(buffer, 32);
			cout << buffer;
		}
	}
}