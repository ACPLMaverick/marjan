using UnityEngine;
using System;
using System.Collections.Generic;

namespace Network
{
    public class PlayerData
    {
        #region Public

        public int PlayerID;
        public int Points;

        /**
         * This will indicate number of ALL parts that player's snake posess 
         */
        public int PartsCount;

        /**
         * This will indicate number of Bends parts that player's snake posess
         * This must be at least one as snake's head is always considered a bend part
         */
        public int PartsBendsCount;

        /**
         * This will indicate a part id (0 is head) at which collision has occured.
         * If no collision ocurred, this should be -1.
         */
        public int CollisionAtPart;

        /**
         * Position of parts at which the snake is bent, i.e. the direction changes.
         * Snake's head is always considered a bent part.
         */
        public Vector2[] PartsBentPositions;

        /**
         * Directions (enums) of a bend part.
         */ 
        public SnakeHead.DirectionType[] PartsBentDirections;

        #endregion

        #region Functions Public

        /**
         * Converts PlayerData to byte array which can by sent by network.
         */ 
        public void ToByteArray(ref byte[] bytes, int offset = 0)
        {
            byte[] bPlayerID = BitConverter.GetBytes(PlayerID);
            byte[] bPoints = BitConverter.GetBytes(Points);
            byte[] bPartsCount = BitConverter.GetBytes(PartsCount);
            byte[] bPartsCountBent = BitConverter.GetBytes(PartsBendsCount);
            byte[] bColl = BitConverter.GetBytes(CollisionAtPart);

            for (int i = 0; i < bPlayerID.Length; ++i, ++offset)
            {
                bytes[offset] = bPlayerID[i];
            }

            for (int i = 0; i < bPoints.Length; ++i, ++offset)
            {
                bytes[offset] = bPoints[i];
            }

            for (int i = 0; i < bPartsCount.Length; ++i, ++offset)
            {
                bytes[offset] = bPartsCount[i];
            }

            for (int i = 0; i < bPartsCountBent.Length; ++i, ++offset)
            {
                bytes[offset] = bPartsCountBent[i];
            }

            for (int i = 0; i < bColl.Length; ++i, ++offset)
            {
                bytes[offset] = bColl[i];
            }

            for (int i = 0; i < PartsBendsCount; ++i)
            {
                for (int j = 0; j < 2; ++j)
                {
                    byte[] floatBytes = BitConverter.GetBytes(PartsBentPositions[i][j]);
                    for (int k = 0; k < floatBytes.Length; ++k, ++offset)
                    {
                        bytes[offset] = floatBytes[k];
                    }
                }
            }

            for (int i = 0; i < PartsBendsCount; ++i)
            {
                byte[] dirBytes = BitConverter.GetBytes((int)PartsBentDirections[i]);
                for (int j = 0; j < dirBytes.Length; ++j, ++offset)
                {
                    bytes[offset] = dirBytes[j];
                }
            }
        }

        public byte[] ToByteArray()
        {
            byte[] bytes = new byte[GetByteArraySize()];

            ToByteArray(ref bytes);

            return bytes;
        }

        public int GetByteArraySize()
        {
            return sizeof(int) * (5 + PartsBendsCount) + 2 * sizeof(float) * PartsBendsCount;
        }

        /**
         * Converts byte array to PlayerData object.
         */
        public static PlayerData FromByteArray(byte[] bytes, int offset = 0)
        {
            PlayerData data = new PlayerData();

            data.PlayerID = BitConverter.ToInt32(bytes, offset);
            offset += sizeof(int);
            data.Points = BitConverter.ToInt32(bytes, offset);
            offset += sizeof(int);
            data.PartsCount = BitConverter.ToInt32(bytes, offset);
            offset += sizeof(int);
            data.PartsBendsCount = BitConverter.ToInt32(bytes, offset);
            offset += sizeof(int);
            data.CollisionAtPart = BitConverter.ToInt32(bytes, offset);
            offset += sizeof(int);

            data.PartsBentPositions = new Vector2[data.PartsBendsCount];
            data.PartsBentDirections = new SnakeHead.DirectionType[data.PartsBendsCount];

            for(int i = 0; i < data.PartsBendsCount; ++i)
            {
                for(int j = 0; j < 2; ++j)
                {
                    data.PartsBentPositions[i][j] = BitConverter.ToSingle(bytes, offset);
                    offset += sizeof(float);
                }
            }

            for(int i = 0; i < data.PartsBendsCount; ++i)
            {
                data.PartsBentDirections[i] = (SnakeHead.DirectionType)BitConverter.ToInt32(bytes, offset);
                offset += sizeof(int);
            }

            return data;
        }

        #endregion
    }
}
