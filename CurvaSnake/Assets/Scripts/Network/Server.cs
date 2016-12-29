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

        #endregion

        #region Properties

        #endregion

        #region Protected

        protected internal class PlayerConnectionInfo
        {
            public readonly Socket Socket;
            public readonly IPEndPoint EndP;
            public PlayerData NewestData;
            public PlayerData PreviousData;
            public float TimeStamp;
            public bool NeedToMulticast;

            public PlayerConnectionInfo(Socket sck, IPEndPoint ep)
            {
                Socket = sck;
                EndP = ep;
                TimeStamp = 0;
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

            foreach (KeyValuePair<int, PlayerConnectionInfo> pair in _players)
            {
                if (pair.Value.NeedToMulticast)
                {
                    pair.Value.NeedToMulticast = false;
                    MulticastPlayerData(pair.Key, pair.Value.NewestData);
                }
            }
        }

        #endregion

        #region Functions Protected

        protected override bool ReceivePacket(IAsyncResult data, Packet pck, IPEndPoint remoteEndPoint)
        {
            if(!base.ReceivePacket(data, pck, remoteEndPoint))
            {
                return false;
            }

            // process sent player data
            if(pck.ControlSymbol == SYMBOL_DTA)
            {
                int playerID = GetPlayerIDFromPacket(pck);
                PlayerConnectionInfo info;
                _players.TryGetValue(playerID, out info);

                if(info != null)
                {
                    AckPacket(pck, info.Socket, info.EndP, null);

                    if(pck.TimeStamp > info.TimeStamp)
                    {
                        info.TimeStamp = pck.TimeStamp;
                        info.PreviousData = info.NewestData;
                        info.NewestData = pck.PData;
                        info.NeedToMulticast = true;
                    }
                }
                else
                {
                    Debug.LogWarning("Server: Sent data of not logged player.");
                }
            }

            // login new player
            if(pck.ControlSymbol == SYMBOL_LOG)
            {
                // check if player already connected
                foreach(KeyValuePair<int, PlayerConnectionInfo> pair in _players)
                {
                    if(pair.Value.EndP.Address.Equals(remoteEndPoint.Address))
                    {
                        return true;
                    }
                }

                int newId = _players.Count + 1;

                Socket pSocket = new Socket(AddressFamily.InterNetwork, SocketType.Dgram, ProtocolType.Udp);
                remoteEndPoint.Port = Client.CLIENT_PORT_LISTEN;
                PlayerConnectionInfo pcinfo = new PlayerConnectionInfo(pSocket, remoteEndPoint);
                _players.Add(newId, pcinfo);

                AckPacket(pck, pcinfo.Socket, pcinfo.EndP, BitConverter.GetBytes(newId));
            }

            return true;
        }

        protected void MulticastPlayerData(int playerID, PlayerData data)
        {
            Packet packet = new Packet();
            packet.ControlSymbol = SYMBOL_DTA;
            packet.PData = data;

            foreach (KeyValuePair<int, PlayerConnectionInfo> pair in _players)
            {
                //if (pair.Key == playerID)
                //    continue;

                SendPacket(packet, pair.Value.Socket, pair.Value.EndP);
            }
        }

        #endregion
    }
}