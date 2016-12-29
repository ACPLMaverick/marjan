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
            public readonly Socket Socket;
            public readonly IPEndPoint EndP;

            public PlayerConnectionInfo(Socket sck, IPEndPoint ep)
            {
                Socket = sck;
                EndP = ep;
            }
        }

        protected Dictionary<int, PlayerConnectionInfo> _players = new Dictionary<int, PlayerConnectionInfo>();

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

        protected override bool ReceivePacket(IAsyncResult data, Packet pck, IPEndPoint remoteEndPoint)
        {
            if(!base.ReceivePacket(data, pck, remoteEndPoint))
            {
                return false;
            }

            // login new player
            if(pck.RawData[0] == SYMBOL_LOG)
            {
                // check if player already connected
                foreach(KeyValuePair<int, PlayerConnectionInfo> pair in _players)
                {
                    if(pair.Value.EndP.Address.Equals(remoteEndPoint.Address))
                    {
                        return false;
                    }
                }

                int newId = _players.Count + 1;

                Socket pSocket = new Socket(AddressFamily.InterNetwork, SocketType.Dgram, ProtocolType.Udp);
                remoteEndPoint.Port = Client.CLIENT_PORT_LISTEN;
                PlayerConnectionInfo pcinfo = new PlayerConnectionInfo(pSocket, remoteEndPoint);
                _players.Add(newId, pcinfo);

                Packet packet = new Packet();
                packet.ControlSymbol = SYMBOL_ACK;
                packet.PacketID = pck.PacketID;
                packet.AdditionalData = new byte[sizeof(int)];

                Array.Copy(BitConverter.GetBytes(newId), packet.AdditionalData, sizeof(int));

                SendPacket(packet, pcinfo.Socket, pcinfo.EndP, true);
            }

            return true;
        }

        protected void AckPacket(Packet pckToAck, PlayerConnectionInfo player, byte[] dataToSend)
        {

        }

        #endregion
    }
}