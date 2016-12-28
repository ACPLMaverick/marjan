using UnityEngine;
using UnityEngine.Events;
using System;
using System.Net;
using System.Net.Sockets;
using System.Security.Cryptography;

namespace Network
{
    public class Packet
    {
        public byte ControlSymbol { get; set; }
        public int PacketID { get; private set; }
        public int Checksum { get; private set; }
        public PlayerData PData { get; set; }

        public byte[] RawData { get { return _rawData; } }

        private byte[] _rawData;

        public void CreateRawData()
        {
            PacketID = BitConverter.ToInt32(BitConverter.GetBytes(Time.time), 0);
            PacketID = PacketID << 8;
            if(PData != null)
            {
                PacketID |= (byte)PData.PlayerID;
            }

            Checksum = 0;

            int size = 2 * sizeof(int) + 1;
            if(PData != null)
            {
                size += PData.GetByteArraySize();
            }
            _rawData = new byte[size];

            _rawData[0] = ControlSymbol;
            byte[] pidArray = BitConverter.GetBytes(PacketID);
            Array.Copy(pidArray, 0, _rawData, 1, sizeof(int));

            if(PData != null)
            {
                PData.ToByteArray(ref _rawData, 9);

                MD5 md5 = MD5.Create();
                Checksum = BitConverter.ToInt32(md5.ComputeHash(_rawData, 9, PData.GetByteArraySize()), 0);
            }

            byte[] csArray = BitConverter.GetBytes(Checksum);
            Array.Copy(csArray, 0, _rawData, 5, sizeof(int));
        }

        public static Packet FromRawData(byte[] rawData)
        {
            Packet pck = new Packet();
            pck._rawData = rawData;

            pck.ControlSymbol = pck._rawData[0];
            pck.PacketID = BitConverter.ToInt32(pck._rawData, 1);
            pck.Checksum = BitConverter.ToInt32(pck._rawData, 5);
            pck.PData = PlayerData.FromByteArray(pck._rawData, 9);

            return pck;
        }

        public bool CheckDataIntegrity()
        {
            if(Checksum == 0)
            {
                return true;
            }
            else
            {
                MD5 md5 = MD5.Create();
                int checksum = BitConverter.ToInt32(md5.ComputeHash(_rawData, 9, PData.GetByteArraySize()), 0);

                return checksum == Checksum;
            }
        }
    }
}
