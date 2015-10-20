using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;

namespace Zad2
{
    public abstract class User
    {
        #region constants

        protected const int SLEEP_AFTER_ACTION_MS = 500;

        #endregion

        #region enums

        public enum UserType
        {
            RANDOM,
            FIXED,
            NONE
        };

        #endregion

        #region variables

        protected Database db;
        protected Thread myThread;
        protected string typeStr;

        #endregion

        #region properties

        public uint ID { get; protected set; }
        public int AddValue { get; protected set; }
        public UserType Type { get; protected set; }

        #endregion

        #region methods

        public User(uint id, int addValue, Database db)
        {
            this.ID = id;
            this.AddValue = addValue;
            this.db = db;
        }

        public void Run()
        {
            myThread = new Thread(new ThreadStart(ModifyValue));
            myThread.Start();

            while (!myThread.IsAlive) ;
        }

        protected void ModifyValue()
        {
            while(true)
            {
                // select value we want to modify
                int valIndex = SelectValue();
                //Console.WriteLine("User " + typeStr + String.Format("{0}", ID) + " modifies item " + String.Format("{0}", valIndex));

                // modify an element in database, what takes some time
                db.ModifyItem(valIndex, AddValue, this);

                if(Type == UserType.FIXED)
                {
                    ++db.FCtr;
                }
                else if(Type == UserType.RANDOM)
                {
                    ++db.RCtr;
                }
            }
        }

        protected virtual int SelectValue()
        {
            return 0;
        }

        #endregion
    }
}
