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
        private const int TRIALS_COUNT = 10;
        private sbyte[] PATTERN = new sbyte[] { 1, 0, 1, 0, 1, 0, 1, 0 };

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
            Network.Initialize(NEURON_COUNT, RECURSION_COUNT, PATTERN);
        }

        public bool Run()
        {
            if (trialCounter >= TRIALS_COUNT)
                return false;

            // generate similar pattern to the one saved in network and send it to the network in attempt to correct it.

            // run neural network
            sbyte[] fPattern = new sbyte[] { 0, 0, 1, 0, 1, 0, 1, 1 };

            System.Console.WriteLine("Controller: Start neural network. Trial #" + (trialCounter + 1).ToString());
            System.Console.WriteLine("Controller: Pattern is:       " + PatternToString(PATTERN, NEURON_COUNT));
            System.Console.WriteLine("Controller: False Pattern is: " + PatternToString(fPattern, NEURON_COUNT));

            sbyte[] retVal = Network.Run(fPattern);

            if (retVal != null)
            {
                System.Console.WriteLine("Controller: Finished trial #" + (trialCounter + 1).ToString() + ". Pattern is:    " + PatternToString(retVal, NEURON_COUNT));
                System.Console.WriteLine();
            }
            else
            {
                System.Console.WriteLine("Controller: Failed trial #" + (trialCounter + 1).ToString());
            }

            ++trialCounter;
            Network.Restart();

            return true;
        }

        private string PatternToString(sbyte[] pattern, int length)
        {
            string str = "";

            for (int i = 0; i < length; ++i)
            {
                str += pattern[i].ToString();
            }

            return str;
        }

        #endregion
    }
}
