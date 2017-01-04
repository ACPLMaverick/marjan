using UnityEngine;
using UnityEngine.Events;
using System;
using System.Net;
using System.Net.Sockets;
using System.Security.Cryptography;
using System.Runtime.Serialization.Formatters.Binary;
using System.IO;

namespace Network
{
    /// <summary>
    /// Network packet class
    /// </summary>
    public class Packet
    {
        private const int HEADER_SIZE = 4 * sizeof(int) + 4 + sizeof(float);

        public byte ControlSymbol { get; set; }
        public byte PartialFlag { get; private set; }
        public byte PDataFlag { get; private set; }
        public byte ADataFlag { get; private set; }
        public int TotalLength { get; private set; }
        public int Offset { get; private set; }
        public int PacketID { get { return _packetID; } set { _packetID = value; } }
        public int Checksum { get; private set; }
        public float TimeStamp { get; private set; }
        public PlayerData PData { get; set; }
        public byte[] AdditionalData { get; set; }

        public byte[] RawData { get { return _rawData; } }

        private byte[] _rawData;
        private int _packetID = 0;

        /// <summary>
        /// Player's packet is being serialized to byte type
        /// </summary>
        public void CreateRawData()
        {
            if(PacketID == 0)
            {
                PacketID = BitConverter.ToInt32(BitConverter.GetBytes(GameController.TimeSeconds * 1000.0f), 0);
                PacketID = PacketID << 8;
                if(PData != null)
                {
                    PacketID |= (byte)PData.PlayerID;
                }
            }

            TimeStamp = GameController.TimeSeconds;

            Checksum = 0;
            Offset = 0;
            PDataFlag = 0;
            ADataFlag = 0;

            TotalLength = HEADER_SIZE;
            int dataSize = 0;
            int pDataSize = 0;
            if(PData != null)
            {
                TotalLength += PData.GetByteArraySize();
                dataSize += PData.GetByteArraySize();
                pDataSize = PData.GetByteArraySize();

                PDataFlag = 1;
            }
            if(AdditionalData != null)
            {
                TotalLength += AdditionalData.Length;
                dataSize += AdditionalData.Length;

                ADataFlag = 1;
            }

            _rawData = new byte[TotalLength];

            _rawData[0] = ControlSymbol;
            _rawData[1] = PartialFlag;
            _rawData[2] = PDataFlag;
            _rawData[3] = ADataFlag;

            byte[] tlArray = BitConverter.GetBytes(TotalLength);
            byte[] ofArray = BitConverter.GetBytes(Offset);
            byte[] pidArray = BitConverter.GetBytes(PacketID);
            byte[] tsArray = BitConverter.GetBytes(TimeStamp);

            Array.Copy(tlArray, 0, _rawData, 4, sizeof(int));
            Array.Copy(ofArray, 0, _rawData, 8, sizeof(int));
            Array.Copy(pidArray, 0, _rawData, 12, sizeof(int));
            Array.Copy(tsArray, 0, _rawData, 20, sizeof(float));

            if(PData != null)
            {
                PData.ToByteArray(ref _rawData, HEADER_SIZE);
            }

            if(AdditionalData != null)
            {
                Array.Copy(AdditionalData, 0, _rawData, HEADER_SIZE + pDataSize, AdditionalData.Length);
            }

            if(PData != null || AdditionalData != null)
            {
                MD5 md5 = MD5.Create();
                Checksum = BitConverter.ToInt32(md5.ComputeHash(_rawData, HEADER_SIZE, dataSize), 0);
                md5.Clear();
            }

            byte[] csArray = BitConverter.GetBytes(Checksum);
            Array.Copy(csArray, 0, _rawData, 16, sizeof(int));
        }

        /// <summary>
        /// Packet deserialization
        /// </summary>
        /// <param name="rawData">Array of bytes containing received packet data</param>
        /// <returns></returns>
        public static Packet FromRawData(byte[] rawData)
        {
            Packet pck = new Packet();

            pck.TotalLength = BitConverter.ToInt32(rawData, 4);

            pck._rawData = new byte[pck.TotalLength];
            Array.Copy(rawData, pck._rawData, pck.TotalLength);

            pck.ControlSymbol = pck._rawData[0];
            pck.PartialFlag = pck._rawData[1];
            pck.PDataFlag = pck._rawData[2];
            pck.ADataFlag = pck._rawData[3];
            pck.Offset = BitConverter.ToInt32(pck._rawData, 8);
            pck.PacketID = BitConverter.ToInt32(pck._rawData, 12);
            pck.Checksum = BitConverter.ToInt32(pck._rawData, 16);
            pck.TimeStamp = BitConverter.ToSingle(pck._rawData, 20);

            int pDataOffset = HEADER_SIZE;
            if(pck.PDataFlag != 0)
            {
                pck.PData = PlayerData.FromByteArray(pck._rawData, pDataOffset);
                pDataOffset += pck.PData.GetByteArraySize();
            }
            if(pck.ADataFlag != 0)
            {
                int aDataLength = pck._rawData.Length - pDataOffset;
                pck.AdditionalData = new byte[aDataLength];
                Array.Copy(pck._rawData, pDataOffset, pck.AdditionalData, 0, aDataLength);
            }

            return pck;
        }

        /// <summary>
        /// Checks packet's checksum
        /// </summary>
        /// <returns></returns>
        public bool CheckDataIntegrity()
        {
            if(Checksum == 0)
            {
                return true;
            }
            else
            {
                MD5 md5 = MD5.Create();
                int checksum = BitConverter.ToInt32(md5.ComputeHash(_rawData, HEADER_SIZE, _rawData.Length - HEADER_SIZE), 0);
                md5.Clear();

                return checksum == Checksum;
            }
        }

        /// <summary>
        /// Serializes and adds additional data to packet into proper field
        /// </summary>
        /// <param name="data">Data to serialize</param>
        public void AddAdditionalData(object data)
        {
            BinaryFormatter bf = new BinaryFormatter();
            using (MemoryStream ms = new MemoryStream())
            {
                bf.Serialize(ms, data);
                byte[] bytes = ms.ToArray();

                int adStartIndex = 0;
                if(AdditionalData == null)
                {
                    AdditionalData = new byte[bytes.Length];
                }
                else
                {
                    adStartIndex = AdditionalData.Length;
                    byte[] newAdditionalData = new byte[AdditionalData.Length + bytes.Length];
                    AdditionalData = newAdditionalData;
                }
                Array.Copy(bytes, 0, AdditionalData, adStartIndex, bytes.Length);
            }
        }
    }
}
