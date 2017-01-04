using UnityEngine;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System;
using UnityEngine.Events;

namespace Network
{
    public class Server : Transceiver
    {
        #region Const

        public static readonly int ADDRESS_LOCAL = Utility.GetAddressAsInt(IPAddress.Loopback);
        public const float PLAYER_TIMEOUT_SECONDS = 5.0f;
        public const float PLAYER_PACKAGE_TIME_OFFSET_MIN = 0.0001f;
        public const float PLAYER_PACKAGE_TIME_OFFSET_MAX = 1.0f;
        public const int PORT_SEND = 2301;
        public const int PORT_LISTEN = 2302;

        public const float FRUIT_GENERATE_DELAY_MIN = 1.0f;
        public const float FRUIT_GENERATE_DELAY_MAX = 6.0f;

        #endregion

        #region Properties

        #endregion

        #region Events

        public class ClientEventVector2 : UnityEvent<Vector2> { }

        public ClientEventVector2 EventAddApple = new ClientEventVector2();

        #endregion

        #region Protected

        protected internal class PlayerConnectionInfo
        {
            public readonly Socket Socket;
            public readonly IPEndPoint EndP;
            public PlayerData NewestData;
            public PlayerData PreviousData;
            public float TimeStamp;
            public bool NeedToMulticast = false;
            public float LastReceiveTime;
            public float NewReceiveTime;

            public PlayerConnectionInfo(Socket sck, IPEndPoint ep)
            {
                Socket = sck;
                EndP = ep;
                TimeStamp = 0;
                LastReceiveTime = 0.0f;
                NewReceiveTime = 0.0f;
            }
        }

        protected Dictionary<int, PlayerConnectionInfo> _players = new Dictionary<int, PlayerConnectionInfo>();
        protected Dictionary<int, PlayerConnectionInfo> _playersPending = new Dictionary<int, PlayerConnectionInfo>();

        protected float _timer = 0.0f;

        protected float _appleTimer = 5.0f;
        protected Transform _fruitAreaMin;
        protected Transform _fruitAreaMax;
        protected Vector2 _applePosition;

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

            _fruitAreaMin = GameObject.Find("GameController/FruitAreaMin").GetComponent<Transform>();
            _fruitAreaMax = GameObject.Find("GameController/FruitAreaMax").GetComponent<Transform>();
        }

        protected override void Update()
        {
            base.Update();

            _timer += Time.deltaTime;

            // calculate collisions with other players
            foreach(KeyValuePair<int, PlayerConnectionInfo> pair in _players)
            {
                int collisionID, otherPlayerID, otherPlayerCollisionID;
                CalculateCollisionWithPlayers(pair, out collisionID, out otherPlayerID, out otherPlayerCollisionID);

                if(collisionID != -1)
                {
                    // tell that player that he had a collision
                    Packet colPacket = new Packet();
                    colPacket.ControlSymbol = SYMBOL_COL;
                    colPacket.PacketID = pair.Key;
                    colPacket.PData = pair.Value.NewestData;
                    colPacket.PData.CollisionAtPart = collisionID;
                    SendPacket(colPacket, pair.Value.Socket, pair.Value.EndP);

                    // tell other players that this player has lost
                    Packet disPacket = new Packet();
                    disPacket.ControlSymbol = SYMBOL_PDN;
                    disPacket.PacketID = pair.Key;
                    MulticastPacket(disPacket, pair.Key);

                    // remove that player
                    _players.Remove(pair.Key);
                    break;
                }
            }

            // send all current player data
            foreach (KeyValuePair<int, PlayerConnectionInfo> pair in _players)
            {
                if (pair.Value.NeedToMulticast)
                {
                    pair.Value.NeedToMulticast = false;
                    MulticastPlayerData(pair.Key, pair.Value.NewestData);
                }
            }

            if (_appleTimer <= 0.0f) // generate new fruit now
            {
               _applePosition = new Vector2
                    (
                        UnityEngine.Random.Range(_fruitAreaMin.position.x, _fruitAreaMax.position.x),
                        UnityEngine.Random.Range(_fruitAreaMin.position.y, _fruitAreaMax.position.y)
                    );
                _appleTimer = 5.0f;

                EventAddApple.Invoke(_applePosition);

                int appleID = BitConverter.ToInt32(BitConverter.GetBytes(Time.time), 0);

                foreach (KeyValuePair<int, PlayerConnectionInfo> pair in _players)
                {
                    MulticastApplePosition(pair.Key, appleID, _applePosition);
                }
            }
            else // decrement the timer
            {
                _appleTimer -= Time.deltaTime;
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
            if (pck.ControlSymbol == SYMBOL_DTA)
            {
                int playerID = GetPlayerIDFromPacket(pck);
                PlayerConnectionInfo info;
                _players.TryGetValue(playerID, out info);

                if (info != null)
                {
                    // sprawdzanie czy pakiety od gracza są otrzymywane za szybko
                    info.NewReceiveTime = _timer;
                    if(info.NewReceiveTime != 0.0f && info.LastReceiveTime != 0.0f)
                    {
                        float diff = info.NewReceiveTime - info.LastReceiveTime;
                        if (diff > PLAYER_PACKAGE_TIME_OFFSET_MAX)
                        {
                            print("Server: Player at ID " + playerID.ToString() + " is sending packets TOO SLOW.");
                            DropPlayer(playerID);
                        }
                        else if (diff < PLAYER_PACKAGE_TIME_OFFSET_MIN)
                        {
                            print("Server: Player at ID " + playerID.ToString() + " is sending packets TOO FAST.");
                            DropPlayer(playerID);
                        }
                    }
                    info.LastReceiveTime = info.NewReceiveTime;
                    ////////////////

                    AckPacket(pck, info.Socket, info.EndP, null);

                    if (pck.TimeStamp > info.TimeStamp)
                    {
                        info.TimeStamp = pck.TimeStamp;
                        info.PreviousData = info.NewestData;
                        info.NewestData = pck.PData;
                        info.NeedToMulticast = true;
                    }
                }
                else
                {
                    Debug.LogWarning("Server: Received data of not logged player.");
                }
            }

            // collision drop
            else if(pck.ControlSymbol == SYMBOL_COL)
            {
                int playerID = GetPlayerIDFromPacket(pck);
                PlayerConnectionInfo info;
                _players.TryGetValue(playerID, out info);

                if(info != null)
                {
                    Debug.Log("Server: Dropping player " + playerID.ToString());
                    // tell other players that this player has lost
                    Packet disPacket = new Packet();
                    disPacket.ControlSymbol = SYMBOL_PDN;
                    disPacket.PacketID = playerID;
                    MulticastPacket(disPacket, playerID);

                    // remove that player
                    _players.Remove(playerID);
                }
            }

            // login new player
            else if (pck.ControlSymbol == SYMBOL_LOG)
            {
                // check if player already connected
                foreach (KeyValuePair<int, PlayerConnectionInfo> pair in _players)
                {
                    if (pair.Value.EndP.Address.Equals(remoteEndPoint.Address))
                    {
                        Debug.LogWarning("Server: Already connected player is connecting.");
                        return true;
                    }
                }

                // check if player is already pending
                foreach (KeyValuePair<int, PlayerConnectionInfo> pair in _playersPending)
                {
                    if (pair.Value.EndP.Address.Equals(remoteEndPoint.Address))
                    {
                        Debug.LogWarning("Server: Already pending connected player is connecting.");
                        return true;
                    }
                }

                int newId = _players.Count + 1;
                Debug.Log("Server: Assigning new id: " + newId.ToString());

                Socket pSocket = new Socket(AddressFamily.InterNetwork, SocketType.Dgram, ProtocolType.Udp);
                remoteEndPoint.Port = Client.CLIENT_PORT_LISTEN;
                PlayerConnectionInfo pcinfo = new PlayerConnectionInfo(pSocket, remoteEndPoint);
                _playersPending.Add(newId, pcinfo);

                AckPacket(pck, pcinfo.Socket, pcinfo.EndP, BitConverter.GetBytes(newId));


                // inform other players that a new one has logged in
                Packet info = new Packet();
                info.ControlSymbol = SYMBOL_PCN;
                info.PacketID = newId;

                MulticastPacket(info, newId);
            }

            // pending player sends his regards
            else if(_playersPending.Count != 0 && pck.ControlSymbol == SYMBOL_ACK && pck.AdditionalData != null)
            {
                int ackID = BitConverter.ToInt32(pck.AdditionalData, 0);

                PlayerConnectionInfo pInfo = null;
                _playersPending.TryGetValue(ackID, out pInfo);
                if(pInfo != null)
                {
                    // tell that player about all connected players on server
                    foreach (KeyValuePair<int, PlayerConnectionInfo> pairInter in _players)
                    {
                        Packet oldPlayersPacket = new Packet();
                        oldPlayersPacket.ControlSymbol = SYMBOL_PCN;
                        oldPlayersPacket.PacketID = pairInter.Key;
                        SendPacket(oldPlayersPacket, pInfo.Socket, pInfo.EndP);
                    }

                    _playersPending.Remove(ackID);
                    _players.Add(ackID, pInfo);
                }
                else
                {
                    Debug.LogWarning("Server: Player sends his ACK for ID but isn't in pending players list.");
                }
            }
            return true;
        }

        protected void DropPlayer(int playerID)
        {
            Debug.Log("Server: DummyDrop of player " + playerID.ToString() + ".");

            //Packet packet = new Packet();
            //packet.ControlSymbol = SYMBOL_PDN;
            //packet.AddAdditionalData(playerID);

            //_players.Remove(playerID);

            //MulticastPacket(packet, playerID);
        }

        protected void MulticastPlayerData(int playerID, PlayerData data)
        {
            Packet packet = new Packet();
            packet.ControlSymbol = SYMBOL_DTA;
            packet.PData = data;

            MulticastPacket(packet, playerID);
        }

        protected void MulticastApplePosition(int playerID, int appleID, Vector2 pos)
        {
            Packet packet = new Packet();
            packet.ControlSymbol = SYMBOL_APL;

            byte[] bytes = new byte[12]; // one int and two floats

            byte[] idBytes = BitConverter.GetBytes(appleID);
            byte[] xBytes = BitConverter.GetBytes(pos.x);
            byte[] yBytes = BitConverter.GetBytes(pos.y);

            for(int i = 0, w = 0; i < 4; ++i, ++w)
            {
                bytes[i] = idBytes[w];
            }
            for (int i = 4, w = 0; i < 8; ++i, ++w)
            {
                bytes[i] = xBytes[w];
            }
            for(int i = 8, w = 0; i < 12; ++i, ++w)
            {
                bytes[i] = yBytes[w];
            }

            packet.AdditionalData = bytes;

            MulticastPacket(packet);
        }

        protected void MulticastPacket(Packet packet, int playerIDToOmitt = -1, bool noAck = false)
        {
            foreach (KeyValuePair<int, PlayerConnectionInfo> pair in _players)
            {
                if (pair.Key == playerIDToOmitt)
                    continue;

                SendPacket(packet, pair.Value.Socket, pair.Value.EndP, noAck);
            }
        }

        protected void CalculateCollisionWithPlayers(KeyValuePair<int, PlayerConnectionInfo> player, out int playerCollisionID, out int otherPlayerID, out int otherPlayerCollisionID)
        {
            float offset = 0.001f;
            playerCollisionID = -1;
            otherPlayerID = -1;
            otherPlayerCollisionID = -1;

            if(player.Value.NewestData != null)
            {
                Vector2 ph = player.Value.NewestData.PartsBentPositions[0];
                foreach (KeyValuePair<int, PlayerConnectionInfo> pair in _players)
                {
                    if (/*pair.Key != player.Key && */pair.Value.NewestData != null)
                    {
                        Vector2[] otherBendPositions = pair.Value.NewestData.PartsBentPositions;
                        int otherBendCount = otherBendPositions.Length;
                        for (int i = (pair.Key == player.Key ? 1 : 0); i < otherBendCount - 1; ++i)
                        {
                            Vector2 a = otherBendPositions[i];
                            Vector2 b = otherBendPositions[i + 1];

                            Vector2 min = Vector2.Min(a, b);
                            Vector2 max = Vector2.Max(a, b);

                            // vertical
                            if(Mathf.Abs(a.x - b.x) < offset)
                            {
                                if(ph.y > (min.y - offset) && ph.y < (max.y + offset) && Mathf.Abs(ph.x - a.x) < offset)
                                {
                                    // collision!
                                    playerCollisionID = 0;
                                    otherPlayerCollisionID = -1;
                                    otherPlayerID = pair.Key;
                                }
                            }
                            // horizontal
                            else if(Mathf.Abs(a.y - b.y) < offset)
                            {
                                if (ph.x > (min.x - offset) && ph.x < (max.x + offset) && Mathf.Abs(ph.y - a.y) < offset)
                                {
                                    // collision!
                                    playerCollisionID = 0;
                                    otherPlayerCollisionID = -1;
                                    otherPlayerID = pair.Key;
                                }
                            }
                        }
                    }
                }
            }

        }

        #endregion
    }
}