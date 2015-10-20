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


        #endregion

        #region variables

        private Data[] items;

        #endregion

        #region properties

        public uint ItemCount { get; set; }

        #endregion

        #region methods

        public Database(uint itemCount)
        {
            ItemCount = itemCount;

            items = new Data[ItemCount];

            for(int i = 0; i < ItemCount; ++i)
            {
                items[i] = new Data();
                items[i].Value = 0;
            }
        }

        public int GetItem(int i)
        {
            int dataToRet;
            Monitor.Enter(items[i]);
            dataToRet = items[i].Value;
            Monitor.Exit(items[i]);

            return dataToRet;
        }

        public void SetItem(int i, int val)
        {
            Monitor.Enter(items[i]);

            items[i].Value = val;
            
            Monitor.Exit(items[i]);
        }

        public override string ToString()
        {
            string str = "";

            for (int i = 0; i < ItemCount; ++i )
            {
                str += String.Format("{0}", items[i].Value) + " ";
            }
            str += "\n";

            return str;
        }

        #endregion
    }
}
