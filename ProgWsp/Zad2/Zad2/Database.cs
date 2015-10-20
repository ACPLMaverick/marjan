using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Zad2
{
    public class Database
    {
        #region constants


        #endregion

        #region variables

        private int[] items;

        #endregion

        #region properties

        public uint ItemCount { get; set; }

        #endregion

        #region methods

        public Database(uint itemCount)
        {
            ItemCount = itemCount;

            items = new int[ItemCount];

            for(int i = 0; i < ItemCount; ++i)
            {
                items[i] = 0;
            }
        }

        public int GetItem(uint i)
        {
            return 0;
        }

        public void SetItem(uint i)
        {

        }

        #endregion
    }
}
