using System;
using System.Net;
using System.Net.Sockets;
using System.Threading;

namespace Network
{
    public class ClientIntAsyncResult : IAsyncResult
    {
        int _data;
        bool _completedSynchronously;
        bool _isCompleted;

        public object AsyncState
        {
            get
            {
                return _data;
            }
        }

        public WaitHandle AsyncWaitHandle
        {
            get
            {
                throw new NotImplementedException();
            }
        }

        public bool CompletedSynchronously
        {
            get
            {
                return _completedSynchronously;
            }
        }

        public bool IsCompleted
        {
            get
            {
                return _isCompleted;
            }
        }

        public ClientIntAsyncResult(int data, bool cs, bool c)
        {
            _data = data;
            _completedSynchronously = cs;
            _isCompleted = c;
        }
    }

    public class Client
    {
        #region Const

        public const int CLIENT_PORT_LISTEN = 2303;

        #endregion
        #region Private

        #endregion

        #region Functions Public

        public void Connect(AsyncCallback connected)
        {
            connected(new ClientIntAsyncResult(1, true, true));
        }

        public void SendDataToServer(int id, PlayerData data, AsyncCallback sent)
        {
            
        }

        #endregion
    }
}