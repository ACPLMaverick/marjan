using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;

namespace Zad2
{
    public class Database
    {
        #region constants

        public const int MODIFY_TIME_MIN = 100;
        public const int MODIFY_TIME_MAX = 2000;
        protected const int MAX_WAIT_ON_MONITOR = 10000;

        #endregion

        #region variables

        private int _fCtr;
        private int _rCtr;

        private Data[] items;
        private Random rand;

        private Semaphore semFCtr;
        private Semaphore semRCtr;

        #endregion

        #region properties

        public uint ItemCount { get; set; }
        public int FCtr 
        { 
            get
            {
                return _fCtr;
            }
            set
            {
                semFCtr.WaitOne();
                _fCtr = value;
                semFCtr.Release();
            }
        }
        public int RCtr 
        { 
            get
            {
                return _rCtr;
            }
            set
            {
                semRCtr.WaitOne();
                _rCtr = value;
                semRCtr.Release();
            }
        }

        #endregion

        #region methods

        public Database(uint itemCount)
        {
            ItemCount = itemCount;

            items = new Data[ItemCount];
            rand = new Random();

            semFCtr = new Semaphore(1, 1);
            semRCtr = new Semaphore(1, 1);

            for(int i = 0; i < ItemCount; ++i)
            {
                items[i] = new Data();
                items[i].Value = 0;
            }

            FCtr = 0;
            RCtr = 0;
        }

        public void ModifyItem(int rid, int val, User u)
        {
            // usypianie by zasymulować przetwarzanie zasobu
            Thread.Sleep(rand.Next(MODIFY_TIME_MIN, MODIFY_TIME_MAX));
            items[rid].Value += val;
        }

        public override string ToString()
        {
            string str = "";

            for (int i = 0; i < ItemCount; ++i )
            {
                str += items[i].Value.ToString() + " ";
            }
            str += ("\nRandom count: " + RCtr.ToString() + " Fixed count: " + FCtr.ToString());

            return str;
        }

        #endregion
    }
}
