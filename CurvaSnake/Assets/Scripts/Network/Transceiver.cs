using UnityEngine;
using System;
using System.Net;
using System.Net.Sockets;
using System.Collections.Generic;

namespace Network
{
    public abstract class Transceiver : MonoBehaviour
    {
        #region Public

        public const float MAX_SEND_PACKET_WAIT = 0.1f;
        public const int MAX_SEND_PACKET_RETRY_TIMES = 100;

        #endregion

        #region Protected

        protected class PacketData
        {
            public readonly float TimeStamp;
            public float Timer;
            public int RetryTimes;
            public readonly Packet Pck;

            public PacketData(float ts, Packet pa)
            {
                TimeStamp = ts;
                Timer = 0.0f;
                RetryTimes = 0;
                Pck = pa;
            }
        }

        protected List<PacketData> _packetsSent = new List<PacketData>();

        protected Socket _sendSocket;
        protected EndPoint _sendEndPoint;

        protected Socket _receiveSocket;
        protected EndPoint _receiveEndPoint;
        protected byte[] _receiveData = new byte[Server.MAX_PACKET_SIZE];

        #endregion

        #region MonoBehaviours

        // Use this for initialization
        protected virtual void Start()
        {

        }

        // Update is called once per frame
        protected virtual void Update()
        {
            UpdatePacketsSent();
            UpdatePacketsReceived();
        }

        #endregion

        #region Functions Public

        #endregion

        #region Functions Protected

        protected void UpdatePacketsSent()
        {
            int psCount = _packetsSent.Count;
            for(int i = 0; i < psCount; ++i)
            {
                if(_packetsSent[i].RetryTimes > MAX_SEND_PACKET_RETRY_TIMES)
                {
                    _packetsSent.RemoveAt(i);
                    BreakConnection();
                    break;
                }
                else if(_packetsSent[i].Timer > MAX_SEND_PACKET_WAIT)
                {
                    SendPacket(_packetsSent[i].Pck, true);
                    ++_packetsSent[i].RetryTimes;
                }
                else
                {
                    _packetsSent[i].Timer += Time.deltaTime;
                }
            }
        }

        protected void UpdatePacketsReceived()
        {

        }

        protected virtual void BreakConnection()
        {
            Debug.LogError("Connection broken, sent packet was not ack'ed.");
        }

        protected virtual void SendPlayerData(PlayerData pd)
        {

        }

        protected virtual void SendPacket(Packet packet, bool retrying = false)
        {
            packet.CreateRawData();
            int sent = _sendSocket.SendTo(packet.RawData, _sendEndPoint);

            if(!retrying)
            {
                _packetsSent.Add(new PacketData(Time.time, packet));
            }
        }

        protected virtual void ListenerInternal(IAsyncResult data)
        {
            // set up packet from received data
            Packet packet = Packet.FromRawData(_receiveData);
            ReceivePacket(data, packet);
        }

        protected virtual bool ReceivePacket(IAsyncResult data, Packet pck)
        {
            if(!pck.CheckDataIntegrity())
            {
                return false;
            }

            // check for acks for sent packets
            if(pck.RawData[0] == Server.SYMBOL_ACK)
            {
                int snum = _packetsSent.Count;
                for(int i = 0; i < snum; ++i)
                {
                    if(_packetsSent[i].Pck.PacketID == BitConverter.ToInt32(pck.RawData, 1))
                    {
                        _packetsSent.RemoveAt(i);
                        break;
                    }
                }
            }

            return true;
        }

        protected virtual bool ReceivePlayerData(List<Packet> packets)
        {
            return true;
        }

        #endregion

        #region Callbacks

        protected void CbListener(IAsyncResult data)
        {
            Debug.Log("Derp.");
            EndPoint remoteEndPoint = new IPEndPoint(0, 0);
            try
            {
                int bytesRead = _receiveSocket.EndReceiveFrom(data, ref remoteEndPoint);

                ListenerInternal(data);

                _receiveSocket.BeginReceiveFrom(_receiveData, 0, Server.MAX_PACKET_SIZE, SocketFlags.None, ref _receiveEndPoint, CbListener, this);

                Debug.LogFormat("ServerListener: Recieved {0} bytes.", bytesRead);

            }
            catch (SocketException e)
            {
                _receiveSocket.EndReceiveFrom(data, ref remoteEndPoint);
                Debug.LogFormat("ServerListener: {0} {1}", e.ErrorCode, e.Message);
            }
        }

        #endregion
    }

}