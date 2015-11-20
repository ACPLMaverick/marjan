using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Win32;

namespace Zad4
{
    public class Database
    {
        public const int MODIFY_TIME_MIN = 100;
        public const int MODIFY_TIME_MAX = 2000;

        private Random random;

        public int[] availableInputResources;
        public int[] totalResources; //available input + alloc from all processes

        #region methods

        public Database(int itemCount)
        {
            random = new Random();

            availableInputResources = new int[itemCount];
            totalResources = new int[itemCount];
        }

        public void CalculateTotalResources(int arraySize, Client[] clients)
        {
            int[] allocFromClients = new int[arraySize];

            for (int k = 0; k < arraySize; ++k)
            {
                allocFromClients[k] = 0;
                for (int l = 0; l < clients.Length; ++l)
                {
                    allocFromClients[k] += clients[l].currAllocated[k];
                }
            }

            for (int i = 0; i < totalResources.Length; ++i)
            {
                totalResources[i] = availableInputResources[i] + allocFromClients[i];
                Console.Write(totalResources[i]);
            }
            Console.WriteLine();
        }

        public void TakeLoan(int[] values, Client client)
        {
            Monitor.Enter(availableInputResources);

            for (int i = 0; i < values.Length; ++i)
            {
                if (values[i] > availableInputResources[i])
                {
                    Monitor.Wait(availableInputResources);
                }
            }

            for (int i = 0; i < values.Length; ++i)
            {
                availableInputResources[i] += client.currAllocated[i];
                client.finished = true;
            }

            Thread.Sleep(random.Next(MODIFY_TIME_MIN, MODIFY_TIME_MAX));
            Console.WriteLine(client.ID + "accessed");

            for (int i = 0; i < availableInputResources.Length; ++i)
            {
                Console.Write(availableInputResources[i]);
            }
            Console.WriteLine();

            Monitor.PulseAll(availableInputResources);

            Monitor.Exit(availableInputResources);
        }

        #endregion
    }
}
