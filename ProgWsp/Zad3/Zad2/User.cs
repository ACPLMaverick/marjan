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
        protected string typeStr;

        #endregion

        #region properties

        public uint ID { get; protected set; }
        public int AddValue { get; protected set; }
        public UserType Type { get; protected set; }
        public Thread MyThread { get; protected set; }

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
            MyThread = new Thread(new ThreadStart(ModifyValue));
            MyThread.Start();

            while (!MyThread.IsAlive) ;
        }

        protected void ModifyValue()
        {
            while(true)
            {
                // wybór wartości do modyfikacji, zależnie od grupy wątku
                int valIndex = SelectValue();

                // modfyikacja elementu w bazie danych, całą synchronizację obsługuje klasa Database
                db.ModifyItem(valIndex, this);

                // inkrementacja liczników modyfikacji w bazie, tu synchronizacja także jest obsługiwana
                // przez klasę Database, przy pomocy semafora
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
