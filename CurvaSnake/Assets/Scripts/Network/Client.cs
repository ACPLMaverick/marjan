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
        public class ClientEventPlayerIDData : UnityEvent<int, PlayerData> { }
        public class ClientEventIntVector2 : UnityEvent<int, Vector2> { }
        /**
         * Used to notify Game Controller that a new player has connected to the server and it needs to
         * instantiate him locally.
         * Arg - player ID 
         */
        public ClientEventPlayerID EventPlayerConnected;

        /**
         * Used to notify that a player has been disconnected and can safely be destroyed locally
         * Arg - player ID
         */
        public ClientEventPlayerID EventPlayerDisconnected;

        public ClientEventPlayerIDData EventPlayerDataReceived;

        public ClientEventIntVector2 EventAddApple;

        #endregion

        #endregion

        #region Monobehaviours

        protected virtual void Awake()
        {
            EventPlayerConnected = new ClientEventPlayerID();
            EventPlayerDisconnected = new ClientEventPlayerID();
            EventPlayerDataReceived = new ClientEventPlayerIDData();
            EventAddApple = new ClientEventIntVector2();
        }

        #endregion

        #region Private

        private IPAddress _serverAddress;

        private UnityAction<int> _afterConnectingAction;

        private bool _connected = false;
        private int _tempPort;

        #endregion

        #region Functions Public

        public void SetServerAddress(string serverAddress, int port = CLIENT_PORT_LISTEN)
        {
            if (!IPAddress.TryParse(serverAddress, out _serverAddress))
            {
                _serverAddress = new IPAddress(Server.ADDRESS_LOCAL);
            }
            _tempPort = port;
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
            connectPck.ControlSymbol = SYMBOL_LOG;
            SendPacket(connectPck);

            _receiveSocket = new Socket(AddressFamily.InterNetwork, SocketType.Dgram, ProtocolType.Udp);
            _receiveEndPoint = new IPEndPoint(IPAddress.Any, _tempPort);
            _receiveSocket.Bind(_receiveEndPoint);
            _receiveSocket.BeginReceiveFrom(_receiveData, 0, Server.MAX_PACKET_SIZE, SocketFlags.None, ref _receiveEndPoint, CbListener, this);
        }

        /**
         * Splits PlayerData into packets and send each one to the server.
         * Waits for ACK for every sent packet.
         * This method is asynchronous.
         */
        public void SendDataToServer(int id, PlayerData data)
        {
            Packet packet = new Packet();
            packet.ControlSymbol = SYMBOL_DTA;
            packet.PData = data;
            packet.PacketID = id;
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

            // process received data
            if(pck.ControlSymbol == SYMBOL_DTA)
            {
                AckPacket(pck, _sendSocket, _sendEndPoint, null);
                EventPlayerDataReceived.Invoke(GetPlayerIDFromPacket(pck), pck.PData);
            }

            else if(pck.ControlSymbol == SYMBOL_PCN)
            {
                AckPacket(pck, _sendSocket, _sendEndPoint, null);
                int id = GetPlayerIDFromPacket(pck);
                EventPlayerConnected.Invoke(id);
            }

            else if(pck.ControlSymbol == SYMBOL_PDN)
            {
                AckPacket(pck, _sendSocket, _sendEndPoint, null);
                EventPlayerDisconnected.Invoke(GetPlayerIDFromPacket(pck));
            }

            // check for connection ACK
            else if (!_connected && pck.ControlSymbol == SYMBOL_ACK)
            {
                AckPacket(pck, _sendSocket, _sendEndPoint, pck.AdditionalData);

                ConnectAfterAck(BitConverter.ToInt32(pck.AdditionalData, 0));
            }

            // check for apple position
            else if(pck.ControlSymbol == SYMBOL_APL)
            {
                AckPacket(pck, _sendSocket, _sendEndPoint, null);

                Vector2 outVector = new Vector2(0, 0);
                int offset = 0;
                byte[] idBytes = new byte[4];
                byte[] floatBytes = new byte[8];

                for (int i = 0; i < 4; ++i)
                {
                    idBytes[i] = pck.AdditionalData[i];
                }
                for (int i = 4, w = 0; i < 12; ++i, ++w)
                {
                    floatBytes[w] = pck.AdditionalData[i];
                }

                int fruitID = BitConverter.ToInt32(idBytes, 0);

                outVector.x = BitConverter.ToSingle(floatBytes, offset);
                offset += sizeof(float);
                outVector.y = BitConverter.ToSingle(floatBytes, offset);

                //print(outVector);

                EventAddApple.Invoke(fruitID, outVector);
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