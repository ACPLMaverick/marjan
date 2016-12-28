using UnityEngine;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System;

namespace Network
{
    public class Server : Transceiver
    {
        #region Const

        public static readonly int ADDRESS_LOCAL = Utility.GetAddressAsInt(IPAddress.Loopback);
        public const float PLAYER_TIMEOUT_SECONDS = 5.0f;
        public const int PORT_SEND = 2301;
        public const int PORT_LISTEN = 2302;
        public const int MAX_PACKET_SIZE = 1280;
        public const byte SYMBOL_LOG = 0x96;
        public const byte SYMBOL_DTA = 0x69;
        public const byte SYMBOL_ACK = 0x6;
        public const byte SYMBOL_NAK = 0x15;

        #endregion

        #region Properties

        #endregion

        #region Protected

        protected struct PlayerConnectionInfo
        {
            public readonly IPAddress Address;
            public readonly Socket Socket;
            public readonly IPEndPoint EndP;

            public PlayerConnectionInfo(IPAddress ipa, Socket sck, IPEndPoint ep)
            {
                Address = ipa;
                Socket = sck;
                EndP = ep;
            }
        }

        protected Dictionary<int, PlayerConnectionInfo> _players;

        #endregion

        #region MonoBehaviours

        protected virtual void Awake()
        {
            _receiveSocket = new Socket(AddressFamily.InterNetwork, SocketType.Dgram, ProtocolType.Udp);
            _receiveEndPoint = new IPEndPoint(IPAddress.Any, PORT_LISTEN);
            _receiveSocket.Bind(_receiveEndPoint);
            _receiveSocket.BeginReceiveFrom(_receiveData, 0, MAX_PACKET_SIZE, SocketFlags.None, ref _receiveEndPoint, CbListener, this);
        }

        protected override void Start()
        {
            base.Start();
        }

        protected override void Update()
        {
            base.Update();
        }

        #endregion

        #region Functions Protected

        protected override bool ReceivePacket(IAsyncResult data, Packet pck)
        {
            if(!base.ReceivePacket(data, pck))
            {
                return false;
            }

            // login new player
            if(pck.RawData[0] == SYMBOL_LOG)
            {
                int newId = _players.Count + 1;
                //_players.Add(newId);
                
                
            }

            return true;
        }

        protected void AckPacket(Packet pckToAck, PlayerConnectionInfo player, byte[] dataToSend)
        {

        }

        #endregion
    }
}