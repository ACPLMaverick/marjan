using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Threading;
using System.IO;
using System.IO.Ports;
using System.Security.Cryptography;

namespace Zad3
{
    class Program
    {
        const byte SOH = 0x01;
        const byte EOT = 0x04;
        const byte ACK = 0x06;
        const byte NAK = 0x15;
        const byte CAN = 0x18;
        const byte C   = 0x43;
        const byte NUL = 0x00;
        const byte SUB = 0x1A;          // do dopełnienia danych w 128-bajtowym pakiecie

        static string filePath = "C:\\tekst.txt";
        static string portName = "COM1";

        static void Main(string[] args)
        {
            Console.WriteLine("XMODEM");
            char choiceFunc = giveFuncionality();
            if (choiceFunc != '1' && choiceFunc != '2') throw new ArgumentException("Bad option");

            givePortNumber();
            giveFilePath();
            
            if(choiceFunc == '1')
            {
                // TRYB ODBIORNIKA
                SerialPort port = new SerialPort(portName, 9600);
                port.DataBits = 8;
                port.StopBits = StopBits.One;
                port.Parity = Parity.None;
                connect(port);
                File.WriteAllBytes(filePath, Decrypt128(recieve(port)));
            }
            else if(choiceFunc == '2')
            {
                // TRYB NADAJNIKA
                SerialPort port = new SerialPort(portName, 9600);
                port.DataBits = 8;
                port.StopBits = StopBits.One;
                port.Parity = Parity.None;
                connect(port);
                if(File.Exists(filePath))
                {
                    send(port, Encrypt128(File.ReadAllBytes(filePath)));
                }
                else
                {
                    throw new IOException("File does not exist.");
                }
            }
            Console.WriteLine("Program finished.");
            Console.ReadKey();
        }

        static char giveFuncionality()
        {
            Console.WriteLine("Select program funcionality:");
            Console.WriteLine("1: Reciever");
            Console.WriteLine("2: Sender");
            ConsoleKeyInfo choiceFunc = Console.ReadKey();
            Console.WriteLine();
            return choiceFunc.KeyChar;
        }

        static void givePortNumber()
        {
            Console.WriteLine("Give COM port number: ");
            char number = (Console.ReadKey()).KeyChar;
            Console.WriteLine();
            if (number > 47 && number < 58) portName = "COM" + number.ToString();
            else Console.WriteLine("Incorrect number given. Using default port " + portName + ".");
        }

        static void giveFilePath()
        {
            Console.WriteLine("Give file path: ");
            string myPath = Console.ReadLine();
            if (myPath.Length > 3) filePath = myPath;
            else Console.WriteLine("Incorrect path given. Using default path " + filePath);
        }

        static byte[] Encrypt128(byte[] input)
        {
            int checkModulo = input.Length % 128;
            int packetCount = Convert.ToInt32(Math.Ceiling(Convert.ToDouble(input.Length) / 128));
            byte[] data = new byte[packetCount * 128];
            // jeżeli plik ma rozmiar o wielokrotności 128 bajtów
            if(checkModulo == 0)
            {
                for(int i = 0; i<(packetCount*128); i++)
                {
                    data[i] = input[i];
                }
            }
                // jeżeli nie - dopełniamy znakiem sub do 128
            else
            {
                for(int i=0;i<input.Length; i++)
                {
                    data[i] = input[i];
                }
                for(int i = input.Length; i< data.Length; i++)
                {
                    data[i] = SUB;
                }
            }
            return data;
        }

        static byte[] Decrypt128(byte[] input)
        {
            int size = 0;
            while(size<input.Length)
            {
                if (input[size] == SUB) break;
                size++;
            }
            byte[] data = new byte[size];
            for (int i = 0; i < data.Length; i++ )
            {
                data[i] = input[i];
            }

            return data;
        }

        static void Rewrite(byte[,] dest, byte[] source, byte sizeDest, byte sizeSource)
        {
            for (int i = 0; i < sizeSource; i++) dest[sizeDest, i] = source[i];
        }

        static byte[] Rewrite(byte[,] source, byte sizeSource)
        {
            byte[] dest = new byte[128];
            for (int i = 0; i < 128; i++) dest[i] = source[sizeSource, i];
            return dest;
        }

        static byte NormalSum(byte[] buffer)
        {
            // Zwykła suma kontrolna
            uint sum = 0;
            foreach (byte x in buffer) sum += x;
            Console.WriteLine("UINT CHECKSUM: " + sum.ToString());
            byte returnSum = Convert.ToByte(sum % 256);
            Console.WriteLine("BYTE CHECKSUM: " + returnSum.ToString());
            return returnSum;
        }

        static void portWriteByte(byte data, SerialPort port)
        {
            // zapisywanie do portu 1 bajtu z offsetem 0
            byte[] buffer = new byte[1];
            buffer[0] = data;
            port.Write(buffer, 0, 1);
        }

        static void connect(SerialPort port)
        {
            try
            {
                port.Open();
            }
            catch (Exception e)
            {
                Console.WriteLine("Error: " + e.Message);
            }
            if (port.IsOpen) Console.WriteLine("Connected to port: " + port.PortName);
        }

        static byte[] recieve(SerialPort port)
        {
            byte packet = 1;
            byte[,] bigBuffer = new byte[257, 128];
            byte[] actualBuffer = new byte[128];
            byte[] returnBuffer = new byte[128 * 256];

            Thread.BeginCriticalRegion();
            System.Console.WriteLine("RECIEVER: initialisation started. Sending NAK.");
            for (int i = 0; i < 5; i++ )
            {
                if(port.BytesToRead != 0)
                {
                    Console.WriteLine("RECIEVER: data recieved. Initialisation stopped.");
                    break;
                }
                Console.WriteLine("RECIEVER: Sending NAK...");
                portWriteByte(NAK, port);
                Thread.Sleep(10000);
            }
            Thread.EndCriticalRegion();

            byte myByte = Convert.ToByte(port.ReadByte()); 
            while(myByte != EOT)
            {
                // odczytywanie odebranych rzeczy zgodnie z kolejnością wysłania
                byte packetNumber = Convert.ToByte(port.ReadByte());
                byte packetNumberComplement = Convert.ToByte(port.ReadByte());

                Console.WriteLine("RECIEVER: SOH recieved. Recieving data...");
                // odbieranie bloku danych
                for (int i = 0; i < 128; i++ )
                {
                    actualBuffer[i] = Convert.ToByte(port.ReadByte());
                }
                Console.WriteLine("RECIEVER: Checking data packet integrity...");
                // odbieranie sumy kontrolnej
                byte checksum = Convert.ToByte(port.ReadByte());
                if((checksum == NormalSum(actualBuffer)) && (packetNumber == packet) && (packetNumberComplement == (255-packetNumber)))
                {
                    // jest ok, zapisywanie do dużego bufora i wysyłanie ACK
                    Rewrite(bigBuffer, actualBuffer, packet, 128);
                    portWriteByte(ACK, port);
                    packet++;
                    Console.WriteLine("RECIEVER: Recieved and successfully checked packet #" + packet.ToString() + ". Sending ACK...");
                }
                else
                {
                    // wyliczona suma kontrolna się nie zgadza albo coś
                    portWriteByte(NAK, port);
                    Console.WriteLine("RECIEVER: Recieved packet #" + packet.ToString() + " is faulty. Sending NAK...");
                }
                myByte = Convert.ToByte(port.ReadByte());   // odbieranie SOH/EOT
            }

            // wysyłanie ACK kończącego transmisję
            Console.WriteLine("RECIEVER: Ending transmission. Sending ACK...");
            portWriteByte(ACK, port);

            for (int i = 1, k = 0; i <= 256;i++ )
            {
                for(int j = 0; j<128;j++, k++)
                {
                    returnBuffer[k] = bigBuffer[i,j];
                }
            }

            return returnBuffer;
        }

        static void send(SerialPort port, byte[] bytes)
        {
            Console.WriteLine("SENDER: Waiting for NAK...");
            byte waitingForNAK = 0;
            // oczekiwanie na NAK od odbiorcy
            int counter = 1;
 
            while(true)
            {
                Console.WriteLine("SENDER: Waiting... count " + counter.ToString());
                counter++;
                waitingForNAK = Convert.ToByte(port.ReadByte());
                Console.WriteLine(waitingForNAK.ToString());
                if (waitingForNAK == NAK) break;
                //portWriteByte(NUL, port);
            }
            /*
            Console.WriteLine(waitingForNAK.ToString());

            if(waitingForNAK == C)
            {
                // obsługa C

                while (true)
                {
                    Console.WriteLine("SENDER: Waiting... count " + counter.ToString());
                    counter++;
                    if ((waitingForNAK = Convert.ToByte(port.ReadByte())) > Convert.ToByte(1)) break;
                }
            }
            */

            //if (waitingForNAK == C) isCRC = true;
            if(waitingForNAK != NAK && waitingForNAK != C)
            {
                Console.WriteLine("SENDER: NAK not recieved. Exiting...");
                return;
            }
            byte packet = 1;
            byte[,] bigBuffer = new byte[256, 128];
            byte[] actualBuffer = new byte[128];
            int a = 0;
            int b = 0;
            if(bytes.Length > 0x8000)
            {
                throw new Exception("SENDER: Maximum data load excedeed");
            }

            // dzielenie danych na 128-bajtowe pakiety
            for(int i = 0; i< bytes.Length; i++)
            {
                bigBuffer[b, a] = bytes[i];
                a++;
                if(a>127)
                {
                    a = 0;
                    b++;
                }
            }
            // czyszczenie bufora portu 
            port.ReadExisting();

            // b - ilość pakietów
            for(int i = 0; i < b; i++)
            {
                Console.WriteLine("SENDER: Sending SOH...");
                portWriteByte(SOH, port);       // wysyłanie SOH
                Console.WriteLine("SENDER: Sending packet number...");
                portWriteByte(packet, port);    // wysyłanie numeru pakietu
                Console.WriteLine("SENDER: Sending complementary packet number...");
                portWriteByte(Convert.ToByte(255 - packet), port); //wysyłanie dopełnienia

                Console.WriteLine("SENDER: Sending data...");
                // wysyłanie pakietu danych
                for(int j = 0; j < 128; j++)
                {
                    portWriteByte(bigBuffer[i, j], port);
                }

                //generowanie sumy kontrolnej
           
                    byte checksum = NormalSum(Rewrite(bigBuffer, Convert.ToByte(packet - 1)));
                    Console.WriteLine("SENDER: Sending checksum...");
                    portWriteByte(checksum, port);
                

                // oczekiwanie na ACK
                Console.WriteLine("SENDER: Waiting for ACK...");
                byte myack;
                do
                {
                    myack = Convert.ToByte(port.ReadByte());
                } while (!((myack == ACK) || (myack == NAK)));
                // jeśli ACK to kontynnujemy transmisję, jeśli NAK, to ponawiamy dla tego samego pakietu
                if(myack == ACK)
                {
                    packet++;
                    Console.WriteLine("SENDER: ACK recieved. Continuing transmission.");
                }
                else if(myack == NAK)
                {
                    i--;
                    Console.WriteLine("SENDER: NAK recieved. Repeating transmission.");
                }
            }


            // finalizowanie transmisji
            Console.WriteLine("SENDER: Finishing transmission...");
            bool check = true;
            port.ReadTimeout = 500;
            // wysyłanie EOT i oczekiwanie na ACK od odbiorcy
            while (check)
            {
                portWriteByte(EOT, port);
                Thread.Sleep(500);
                if (port.ReadByte() == ACK) check = false;
            }
            Console.WriteLine("SENDER: ACK recieved. Transmission terminated.");
            port.ReadTimeout = 10000;
        }
    }
}
