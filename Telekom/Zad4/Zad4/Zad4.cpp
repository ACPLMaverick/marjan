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
		// zautomatyzowane wysy�anie komend j�zyka Hayesa w celu nawi�zania po��czenia modemowego
		if (functionality == 1)
		{
			PortSendString("ATM0", HandlePort);		// wy��cz g�o�nik
			PortSend(10, HandlePort);
			PortSend(13, HandlePort);
			WaitForOK(HandlePort);
			PortSendString("ATC1", HandlePort);		// utw�rz fal� no�n�
			PortSend(10, HandlePort);
			PortSend(13, HandlePort);
			WaitForOK(HandlePort);
			PortSendString("ATD55", HandlePort);	// zadzwo� pod nr 55
			PortSend(10, HandlePort);
			PortSend(13, HandlePort);
		}
		else if (functionality == 2)
		{
			PortSendString("ATM0", HandlePort);		// wy��cz g�o�nik
			PortSend(10, HandlePort);
			PortSend(13, HandlePort);
			WaitForOK(HandlePort);
			PortSendString("ATH1", HandlePort);		// podnie� s�uchawk�
			PortSend(10, HandlePort);
			PortSend(13, HandlePort);
			WaitForOK(HandlePort);
			PortSendString("ATA", HandlePort);		// odbierz po��czenie
			PortSend(10, HandlePort);
			PortSend(13, HandlePort);
		}
		else
		{
			cout << "Something went wrong." << endl;
			return(-1);
		}
	}

	while (true)
	{
		// niesko�czona p�tla, w kt�rej przesy�amy wpisane na konsol� znaki do portu
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
	// wysy�anie stringu do portu
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
	// oczekiwanie a� w buforze portu pojawi si� string OK
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
	// otwarcie portu o zadaniej nazwie
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
	// pobiera domy�lne ustawienie z portu przypisanego do HandlePort
	if (GetCommState(HandlePort, &SerialPort) == 0)
	{
		cout << "GetCommState error" << endl;
		CloseHandle(HandlePort);
		return false;
	}

	SerialPort.BaudRate = baudRate; 
	SerialPort.fBinary = TRUE; // w��czenie trybu binarnego
	SerialPort.fParity = TRUE; // w��czenie sprawdzania parzysto�ci
	SerialPort.fOutxCtsFlow = FALSE; // monitorowanie sygna�u CTS (clear-to-send)
	SerialPort.fOutxDsrFlow = FALSE;  // jw dla DSR (data-set-ready)
	SerialPort.fDtrControl = DTR_CONTROL_ENABLE; // jw dla DTR (data-terminal-ready)
	SerialPort.fDsrSensitivity = false; // wy��czenie oddzia�ywania sygna�u DSR na sterownik po��czenia
	SerialPort.fTXContinueOnXoff = true; //transmisja jest kontynuowana po tym jak bufor wej�ciowy przekroczy limit o XOffLim bajt�w i sterownik wy�le XoffChar by zako�czy� odbieranie
	SerialPort.fOutX = false; // wy��czenie kontroli przep�ywu XON/XOFF podczas wysy�ania
	SerialPort.fInX = false; // jw dla odbierania
	SerialPort.fErrorChar = false; // wy��czenie zaznaczania bajt�w z b��dami przez ErrorChary
	SerialPort.fNull = false; // wy��czenie odrzucania zerowych bajt�w (null bytes)
	SerialPort.fRtsControl = RTS_CONTROL_ENABLE; // w��czenie kontroli przep�ywu RTS (request-to-send)
	SerialPort.fAbortOnError = false; // wy��czenie zawieszania transmisji przy b��dzie
	SerialPort.ByteSize = 8;
	SerialPort.Parity = NOPARITY; 
	SerialPort.StopBits = ONESTOPBIT;

	// konfiguruje urz�dzenie zgodnie z ustawieniami w bloku DCB obiektu SerialPort
	if (SetCommState(HandlePort, &SerialPort) == 0)
	{
		cout << "SetCommState error" << endl;
		CloseHandle(HandlePort);
		return false;
	}

	COMMTIMEOUTS SerialPortTimeouts;

	// taka sama operacja jak wy�ej, tyle �e dla Timeout�w portu

	if (GetCommTimeouts(HandlePort, &SerialPortTimeouts) == 0)
	{
		cout << "GetCommTimeouts error" << endl;
		CloseHandle(HandlePort);
		return false;
	}

	SerialPortTimeouts.ReadIntervalTimeout = MAXDWORD;	// maksymalna ilo�� czasu pomi�dzy kolejnymi bajtami, w milisekundach (du�o)
	SerialPortTimeouts.ReadTotalTimeoutMultiplier = 0;	// mno�nik u�ywany do ustalenia ca�kowitego timeoutu dla odczytu
	SerialPortTimeouts.ReadTotalTimeoutConstant = 0;	// sta�a u�ywana do j/w
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
	// odczytanie danej liczby bajt�w z portu
	DWORD recieved = 0;
	for (int i = 0; i < dataSize; i++) data[i] = 0;
	ReadFile(HandlePort, data, dataSize - 1, &recieved, NULL);
	return data;
}

bool PortSend(unsigned char myChar, HANDLE HandlePort)
{
	// wysy�anie 1 bajtu do portu
	DWORD bytesWritten = 0;
	while (bytesWritten == 0) WriteFile(HandlePort, &myChar, 1, &bytesWritten, NULL);
	return true;
}

DWORD WINAPI reciever(LPVOID lpParam)
{
	// definiowanie recievera - co si� dzieje przy odebranych znakach - wypisywanie na ekran
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