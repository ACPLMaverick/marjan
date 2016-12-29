using UnityEngine;
using UnityEngine.Events;
using System;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using System.Collections;

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

    public class Client : Transceiver
    {
        #region Const

        public const int CLIENT_PORT_LISTEN = 2303;

        #endregion

        #region Public

        #region Events
        
        public class ClientEventPlayerID : UnityEvent<int> { }
        /**
         * Used to notify Game Controller that a new player has connected to the server and it needs to
         * instantiate him locally.
         * Arg - player ID 
         */
        public ClientEventPlayerID EventPlayerConnected = new ClientEventPlayerID();

        /**
         * Used to notify that a player has been disconnected and can safely be destroyed locally
         * Arg - player ID
         */
        public ClientEventPlayerID EventPlayerDisconnected = new ClientEventPlayerID();

        #endregion

        #endregion

        #region Private

        private IPAddress _serverAddress;

        private UnityAction<int> _afterConnectingAction;

        private bool _connected = false;

        #endregion

        #region Functions Public

        public void SetServerAddress(string serverAddress)
        {
            if (!IPAddress.TryParse(serverAddress, out _serverAddress))
            {
                _serverAddress = new IPAddress(Server.ADDRESS_LOCAL);
            }
        }

        /**
         * Log on to server and recieve an unique player ID which client can use to communicate.
         * Client sends LOGIN byte and waits for a packet with [ACK, PLAYER_ID] content.
         * If already logged in, NAK will be recieved.
         * This method is asynchronous.
         */
        public void Connect(UnityAction<int> callback)
        {
            _sendSocket = new Socket(AddressFamily.InterNetwork, SocketType.Dgram, ProtocolType.Udp);
            _sendEndPoint = new IPEndPoint(_serverAddress, Server.PORT_LISTEN);
            //_sendSocket.Bind(_sendEndPoint);

            _afterConnectingAction = callback;

            Packet connectPck = new Packet();
            connectPck.ControlSymbol = Server.SYMBOL_LOG;
            SendPacket(connectPck);

            _receiveSocket = new Socket(AddressFamily.InterNetwork, SocketType.Dgram, ProtocolType.Udp);
            _receiveEndPoint = new IPEndPoint(IPAddress.Any, CLIENT_PORT_LISTEN);
            _receiveSocket.Bind(_receiveEndPoint);
            _receiveSocket.BeginReceiveFrom(_receiveData, 0, Server.MAX_PACKET_SIZE, SocketFlags.None, ref _receiveEndPoint, CbListener, this);
        }

        /**
         * Splits PlayerData into packets and send each one to the server.
         * Waits for ACK for every sent packet.
         * This method is asynchronous.
         */
        public void SendDataToServer(PlayerData data)
        {
            Packet packet = new Packet();
            packet.ControlSymbol = Server.SYMBOL_DTA;
            packet.PData = data;
            SendPacket(packet);
        }

        #endregion

        #region Functions Protected

        protected override bool ReceivePacket(IAsyncResult data, Packet pck, IPEndPoint remoteEndPoint)
        {
            if(!base.ReceivePacket(data, pck, remoteEndPoint))
            {
                return false;
            }

            // check for connection ACK
            if (!_connected && pck.RawData[0] == Server.SYMBOL_ACK)
            {
                ConnectAfterAck(BitConverter.ToInt32(pck.AdditionalData, 0));
            }

            return true;
        }

        protected void ConnectAfterAck(int newID)
        {
            _connected = true;

            if(_afterConnectingAction != null)
            {
                _afterConnectingAction.Invoke(newID);
            }
        }

        #endregion

        #region Callbacks

        #endregion
    }
}