using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Zad4
{
    public class Client
    {
        //INPUT
        public int[] maxDemand;
        public int[] currAllocated;

        //WE NEED TO FIND
        public int[] need; //max[i] - alloc[i];

        public Thread MyThread;
        public Database db;
        public uint ID;
        public bool finished = false;

        public Client(int resourceCount, uint ID, Database db)
        {
            this.ID = ID;
            this.db = db;
            maxDemand = new int[resourceCount];
            currAllocated = new int[resourceCount];
            need = new int[resourceCount];

            MyThread = new Thread(CheckLoanAvailability);
        }

        public void Run()
        {
            MyThread.Start();

            while (!MyThread.IsAlive) ;
        }

        public void CalculateNeedArray()
        {
            for (int i = 0; i < need.Length; ++i)
            {
                need[i] = maxDemand[i] - currAllocated[i];
                //Console.Write(need[i]);
            }
            //Console.WriteLine();
        }

        public void CheckLoanAvailability()
        {
            while (!finished)
            {
                db.TakeLoan(need, this);

                if (finished) MyThread.Abort();
            }
        }
    }
}
