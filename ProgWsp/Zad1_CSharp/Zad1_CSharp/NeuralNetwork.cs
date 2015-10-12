using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Zad1_CSharp
{
    /// <summary>
    /// This class is a base for a neural network. It governs every neuron's activity and their synchronization, using a binary semaphore.
    /// </summary>
    public class NeuralNetwork
    {
        #region constants

        private const int TIMEOUT = 10000;

        #endregion

        #region variables

        private AutoResetEvent waitHandle;

        #endregion

        #region properties

        public int NeuronCount { get; private set; }
        public int RecursionSteps { get; private set; }
        public byte[] Pattern { get; private set; } // Pattern length = NeuronCount. Always.

        public Neuron[] Neurons { get; private set; }

        public Semaphore SemRecursionCtr { get; private set; }
        public int RecursionCtr { get; set; }

        #endregion

        #region methods

        public void Initialize(int neuronCount, int recursionSteps, byte[] pattern)
        {
            this.NeuronCount = neuronCount;
            this.RecursionSteps = recursionSteps;
            this.Pattern = pattern;
            this.RecursionCtr = 0;

            this.Neurons = new Neuron[this.NeuronCount];

            for(int i = 0; i < this.NeuronCount; ++i)
            {
                this.Neurons[i] = new Neuron();
            }
            for (int i = 0; i < this.NeuronCount; ++i)
            {
                byte[] tab = new byte[] { 3, 3, 3, 3, 3, 3, 3, 3 };
                tab[i] = 0;
                this.Neurons[i].Initialize(i, NeuronCount, this.Neurons, tab, this);
            }

        }

        public byte[] Run(byte[] falsePattern)
        {
            // run neurons
            for (int i = 0; i < NeuronCount; ++i )
            {
                Neurons[i].StartThread(falsePattern, waitHandle);
            }

            // go to sleep until all neuron threads finish
            for (int i = 0; i < NeuronCount; ++i)
            {
                Neurons[i].Thread.Join(TIMEOUT);
            }


            if(RecursionCtr >= RecursionSteps)
            {
                System.Console.WriteLine("NeuralNetwork: Finished.");

                byte[] ret = new byte[NeuronCount];
                for (int i = 0; i < NeuronCount; ++i )
                {
                    ret[i] = (byte)(Neurons[i].Activation);
                }

                return ret;
            }
            else
            {
                System.Console.WriteLine("NeuralNetwork: Failed. Reason: Timeout.");
                return null;
            }
        }

        public void Shutdown()
        {

        }

        #endregion
    }
}
