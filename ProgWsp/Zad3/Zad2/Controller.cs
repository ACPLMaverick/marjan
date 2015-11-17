using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;

namespace Zad2
{
    public class Controller
    {
        #region constants

        public const int ITEM_COUNT = 8;
        private const int DISPLAY_SLEEP = 100;

        #endregion

        #region variables 

        private int randomCount;
        private int fixedCount;

        #endregion

        #region properties

        public Database MDatabase { get; private set; }
        public User[] MUsersRandom { get; private set; }
        public User[] MUsersFixed { get; private set; }

        #endregion

        #region methods

        public Controller()
        {
            MDatabase = new Database(ITEM_COUNT);
            randomCount = ITEM_COUNT + 1;
            fixedCount = ITEM_COUNT;
            MUsersRandom = new User[randomCount];
            MUsersFixed = new User[fixedCount];

            for (int i = 0; i < randomCount; ++i )
            {
                MUsersRandom[i] = new UserRandom((uint)i, -1, MDatabase);
            }

            for (int i = 0; i < fixedCount; ++i)
            {
                MUsersFixed[i] = new UserFixed((uint)i, 1, MDatabase, i);
            }

            Console.WriteLine("Controller initialized.");
        }

        public void Run()
        {
            Console.WriteLine("Running controller...");

            for (int i = 0; i < randomCount; ++i)
            {
                MUsersRandom[i].Run();
            }

            for (int i = 0; i < fixedCount; ++i)
            {
                MUsersFixed[i].Run();
            }

            while(true)
            {
                Console.Clear();
                Console.WriteLine(MDatabase.ToString());

                Thread.Sleep(DISPLAY_SLEEP);
            }
        }

        #endregion
    }
}
