using UnityEngine;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;

namespace Network
{
    public class Server : MonoBehaviour
    {
        #region Const

        public static readonly int SERVER_ADDRESS_LOCAL = Utility.GetAddressAsInt(IPAddress.Loopback);
        public const int SERVER_PORT_LISTEN = 2302;

        #endregion

        #region Properties

        #endregion

        #region MonoBehaviours

        void Start()
        {

        }

        void Update()
        {

        }

        #endregion
    }
}