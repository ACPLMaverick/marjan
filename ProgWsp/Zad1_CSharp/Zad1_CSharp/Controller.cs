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

        const int NEURON_COUNT = 3;
        const int RECURSION_COUNT = 200;
        const int TRIALS_COUNT = 20;

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
            Network.Initialize(NEURON_COUNT, RECURSION_COUNT, 0x00000000);
        }

        public bool Run()
        {
            if (trialCounter > TRIALS_COUNT)
                return false;

            // generate similar pattern to the one saved in network and send it to the network in attempt to correct it.

            bool retVal = Network.Run(0x00000001);
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
