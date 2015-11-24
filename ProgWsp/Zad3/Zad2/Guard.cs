using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;

namespace Zad2
{
    public class Guard
    {
        #region constants

        #endregion

        #region variables

        private Queue<UserFixed> fixedQueue;
        private Queue<UserRandom> randomQueue;
        private Queue<User> enterQueue;
        private Random rand;

        public AutoResetEvent mre;

        #endregion

        #region properties

        public Data MData { get; private set; }
        public Thread MThread { get; private set; }

        #endregion

        #region methods

        public Guard(Data mData)
        {
            this.MData = mData;
            fixedQueue = new Queue<UserFixed>();
            randomQueue = new Queue<UserRandom>();
            enterQueue = new Queue<User>();
            rand = new Random();
            mre = new AutoResetEvent(false);

            MThread = new Thread(new ThreadStart(Run));
            MThread.Start();

            while (!MThread.IsAlive) ;
        }

        public void AddAccess(User u)
        {
            mre.Set();
            enterQueue.Enqueue(u);
            u.mre.WaitOne();
        }

        private void Run()
        {
            while(true)
            {
                if(enterQueue.Count != 0)
                {
                    User nUser = enterQueue.Dequeue();

                    if(nUser.Type != MData.LastUserAccess)
                    {
                        if(nUser.Type == User.UserType.RANDOM)
                        {
                            ProcessData(nUser);
                        }
                        else if(nUser.Type == User.UserType.FIXED)
                        {
                            ProcessData(nUser);
                        }
                    }
                    else
                    {
                        if (nUser.Type == User.UserType.RANDOM)
                        {
                            randomQueue.Enqueue((UserRandom)nUser);
                        }
                        else if (nUser.Type == User.UserType.FIXED)
                        {
                            fixedQueue.Enqueue((UserFixed)nUser);
                        }
                    }
                }
                if(MData.LastUserAccess == User.UserType.FIXED && randomQueue.Count != 0)
                {
                    User uR = randomQueue.Dequeue();
                    ProcessData(uR);
                }
                if(MData.LastUserAccess == User.UserType.RANDOM && fixedQueue.Count != 0)
                {
                    User uF = fixedQueue.Dequeue();
                    ProcessData(uF);
                }

                if (
                    enterQueue.Count == 0 &&
                    randomQueue.Count == 0 && 
                    fixedQueue.Count == 0
                    )
                {
                    mre.WaitOne();
                }
            }
        }

        private void ProcessData(User u)
        {
            MData.Value += u.AddValue;
            MData.LastUserAccess = u.Type;

            Thread.Sleep(rand.Next(Database.MODIFY_TIME_MIN, Database.MODIFY_TIME_MAX));
            u.mre.Set();
        }

        #endregion
    }
}
