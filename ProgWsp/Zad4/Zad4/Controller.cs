using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Zad4
{
    class Controller
    {
        #region constants

        public const int CLIENT_COUNT = 5;
        public const int RESOURCE_COUNT = 4;

        #endregion

        #region properties

        public Database Bank { get; set; }
        public Client[] Clients { get; set; }
        
        #endregion

        #region methods

        public Controller()
        {
            Bank = new Database(RESOURCE_COUNT);
            Clients = new Client[CLIENT_COUNT];

            ReadFile("availableResources.txt", Bank.availableInputResources, 0);

            for (int i = 0; i < CLIENT_COUNT; ++i)
            {
                Clients[i] = new Client(RESOURCE_COUNT, (uint) i, Bank);
                ReadFile("currentlyAllocated.txt", Clients[i].currAllocated, i);
                ReadFile("maximumResources.txt", Clients[i].maxDemand, i);
                Clients[i].CalculateNeedArray();
            }

            Bank.CalculateTotalResources(RESOURCE_COUNT, Clients);
        }

        public void Run()
        {
            for (int i = 0; i < CLIENT_COUNT; ++i)
            {
                Clients[i].Run();
            }

            while (true)
            {
                Thread.Sleep(100);
            }
        }

        void ReadFile(string fileName, int[] outputArray, int lineNumber)
        {
            string[] lines = System.IO.File.ReadAllLines(fileName);
            for (int j = 0; j < lines[lineNumber].Split(' ').Length; ++j)
            {
                outputArray[j] = int.Parse(lines[lineNumber].Split(' ')[j]);
            }
        }

        #endregion
    }
}
