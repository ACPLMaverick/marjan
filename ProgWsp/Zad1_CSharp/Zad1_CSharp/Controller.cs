using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Zad1_CSharp
{
    /// <summary>
    /// This class is responsible for communicating with user and governing NeuralNetwork's activity.
    /// </summary>
    public class Controller
    {
        #region tempConstants

        public const int NEURON_COUNT = 8;
        private const int RECURSION_COUNT = 1000;
        private const int TRIALS_COUNT = 1;

        #endregion

        #region variables

        private uint trialCounter = 0;

        #endregion

        #region properties

        public NeuralNetwork Network { get; private set; }

        #endregion

        #region methods

        public Controller()
        {

        }

        ~Controller()
        {

        }


        public void Initialize()
        {
            // acquire certain network parameters and its pattern and then send them to the network to initialize it.

            Network = new NeuralNetwork();
            Network.Initialize(NEURON_COUNT, RECURSION_COUNT, new byte[] { 1, 0, 1, 0, 1, 0, 1, 0 });
        }

        public bool Run()
        {
            if (trialCounter >= TRIALS_COUNT)
                return false;

            // generate similar pattern to the one saved in network and send it to the network in attempt to correct it.

            bool retVal = Network.Run(new byte[] { 0, 0, 1, 0, 1, 0, 1, 1 });
            if (!retVal)
                return retVal;

            ++trialCounter;

            return true;
        }

        public void Shutdown()
        {
            Network.Shutdown();
        }

        #endregion
    }
}
