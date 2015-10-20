using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Zad2
{
    public abstract class User
    {
        #region constants

        #endregion

        #region variables

        protected Database db;

        #endregion

        #region properties

        public uint ID { get; protected set; }
        public int AddValue { get; protected set; }

        #endregion

        #region methods

        public User(uint id, int addValue, Database db)
        {
            ID = id;
            AddValue = addValue;
            this.db = db;
        }

        public void Run()
        {

        }

        protected void ModifyValue()
        {

        }

        protected virtual int SelectValue()
        {
            return 0;
        }

        #endregion
    }
}
