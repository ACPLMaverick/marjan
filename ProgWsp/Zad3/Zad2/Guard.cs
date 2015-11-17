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

            MThread = new Thread(new ThreadStart(Run));
            MThread.Start();

            while (!MThread.IsAlive) ;
        }

        public void AddAccess(User u)
        {
            enterQueue.Enqueue(u);
            u.MyThread.Suspend();
        }

        private void Run()
        {
            while(true)
            {
                if(enterQueue.Count != 0)
                {
                    User nUser = enterQueue.Dequeue();


                }
                else if(MData.LastUserAccess == User.UserType.FIXED && randomQueue.Count != 0)
                {

                }
                else if(MData.LastUserAccess == User.UserType.RANDOM && fixedQueue.Count != 0)
                {

                }
            }
        }

        private void ProcessFixed(UserFixed uf)
        {

        }

        private void ProcessRandom(UserRandom ur)
        {

        }

        #endregion
    }
}
