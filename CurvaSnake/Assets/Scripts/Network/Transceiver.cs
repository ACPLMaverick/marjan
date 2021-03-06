﻿using UnityEngine;
using System;
using System.Net;
using System.Net.Sockets;
using System.Collections.Generic;

namespace Network
{
    public abstract class Transceiver : MonoBehaviour
    {
        #region Public

        public const int MAX_PACKET_SIZE = 1280;
        public const byte SYMBOL_LOG = 0x96;
        public const byte SYMBOL_DTA = 0x69;
        public const byte SYMBOL_ACK = 0x6;
        public const byte SYMBOL_CCK = 0x7;
        public const byte SYMBOL_NAK = 0x15;
        public const byte SYMBOL_PCN = 0x1;
        public const byte SYMBOL_PDN = 0x2;
        public const byte SYMBOL_APL = 0x3;
        public const byte SYMBOL_COL = 0x4;

        public const float MAX_SEND_PACKET_WAIT = 0.2f;
        public const int MAX_SEND_PACKET_RETRY_TIMES = 50;

        #endregion

        #region Protected

        protected class PacketData
        {
            public readonly float TimeStamp;
            public float Timer;
            public int RetryTimes;
            public readonly Packet Pck;
            public readonly Socket Sck;
            public readonly EndPoint Ep;

            public PacketData(float ts, Packet pa, Socket sck, EndPoint ep)
            {
                TimeStamp = ts;
                Timer = 0.0f;
                RetryTimes = 0;
                Pck = pa;
                Sck = sck;
                Ep = ep;
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
            for(int i = 0; i < _packetsSent.Count; ++i)
            {
                if(_packetsSent[i].RetryTimes > MAX_SEND_PACKET_RETRY_TIMES)
                {
                    _packetsSent.RemoveAt(i);
                    BreakConnection();
                    break;
                }
                else if(_packetsSent[i].Timer > MAX_SEND_PACKET_WAIT)
                {
                    SendPacket(_packetsSent[i].Pck, _packetsSent[i].Sck, _packetsSent[i].Ep, true);

                    ++_packetsSent[i].RetryTimes;
                    _packetsSent[i].Timer = 0.0f;
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
            Debug.LogWarning("Packet was not ack'ed. Dropping it.");
        }

        protected virtual void SendPlayerData(PlayerData pd)
        {

        }

        protected virtual void SendPacket(Packet packet, bool noAck = false)
        {
            SendPacket(packet, _sendSocket, _sendEndPoint, noAck);
        }

        protected virtual void SendPacket(Packet packet, Socket sck, EndPoint ep, bool noAck = false)
        {
            packet.CreateRawData();
            sck.SendTo(packet.RawData, ep);

            if (!noAck)
            {
                _packetsSent.Add(new PacketData(GameController.TimeSeconds, packet, sck, ep));
            }
        }

        protected virtual void ListenerInternal(IAsyncResult data, IPEndPoint remoteEndPoint)
        {
            // set up packet from received data
            Packet packet = Packet.FromRawData(_receiveData);
            ReceivePacket(data, packet, remoteEndPoint);
        }

        protected virtual bool ReceivePacket(IAsyncResult data, Packet pck, IPEndPoint remoteEndPoint)
        {
            Debug.Log("Received packet from " + remoteEndPoint.Address.ToString() + ":" + remoteEndPoint.Port.ToString() + " with control symbol " + pck.ControlSymbol.ToString() + 
                " and ID " + pck.PacketID.ToString());

            if (!pck.CheckDataIntegrity())
            {
                return false;
            }

            // check for acks for sent packets
            if(pck.ControlSymbol == SYMBOL_ACK)
            {
                Begin:;
                int snum = _packetsSent.Count;
                for(int i = 0; i < snum; ++i)
                {
                    if(_packetsSent[i].Pck.PacketID == pck.PacketID)
                    {
                        _packetsSent.RemoveAt(i);
                        goto Begin;
                    }
                }
            }

            return true;
        }

        protected int GetPlayerIDFromPacket(Packet pck)
        {
            return pck.PacketID & 0x000000FF;
        }

        protected void AckPacket(Packet pck, Socket sck, EndPoint ep, byte[] dataToSend)
        {
            Packet packet = new Packet();
            packet.ControlSymbol = SYMBOL_ACK;
            packet.PacketID = pck.PacketID;

            if (dataToSend != null)
            {
                packet.AdditionalData = new byte[dataToSend.Length];
                Array.Copy(dataToSend, packet.AdditionalData, dataToSend.Length);
            }

            SendPacket(packet, sck, ep, true);
        }

        #endregion

        #region Callbacks

        protected void CbListener(IAsyncResult data)
        {
            EndPoint remoteEndPoint = new IPEndPoint(0, 0);
            try
            {
                _receiveSocket.EndReceiveFrom(data, ref remoteEndPoint);

                _receiveSocket.BeginReceiveFrom(_receiveData, 0, MAX_PACKET_SIZE, SocketFlags.None, ref _receiveEndPoint, CbListener, this);

                ListenerInternal(data, (IPEndPoint)remoteEndPoint);
            }
            catch
            {
                _receiveSocket.EndReceiveFrom(data, ref remoteEndPoint);
            }
        }

        #endregion
    }

}