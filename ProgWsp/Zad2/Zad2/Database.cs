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

        public void ModifyItem(int i, int val, User u)
        {
            // wchodzimy do monitora
            Monitor.Enter(items[i]);

            // jeśli ostatni obsługiwany wątek należał do tej samej kategorii
            // ustępujemy miejsca następnemu wątkowi
            if(u.Type == items[i].LastUserAccess)
            {
                Monitor.Wait(items[i]);
            }

            // usypianie by zasymulować przetwarzanie zasobu
            Thread.Sleep(rand.Next(MODIFY_TIME_MIN, MODIFY_TIME_MAX));
            items[i].Value += val;

            // jeśli jesteśmy "nowym" wątkiem, budzimy ten poprzedni, uśpiony
            if (u.Type != items[i].LastUserAccess)
            {
                items[i].LastUserAccess = u.Type;
                Monitor.Pulse(items[i]);
            }
            
            // wychodzimy z monitora
            Monitor.Exit(items[i]);
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
