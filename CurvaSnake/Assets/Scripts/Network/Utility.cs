using System;
using System.Net;
using System.Net.Sockets;

namespace Network
{
    public static class Utility
    {
        public static int GetAddressAsInt(IPAddress address)
        {
            return BitConverter.ToInt32(address.GetAddressBytes(), 0);
        }

        public static IPAddress GetAddressFromInt(int addr)
        {
            return new IPAddress(BitConverter.GetBytes(addr));
        }
    }
}
