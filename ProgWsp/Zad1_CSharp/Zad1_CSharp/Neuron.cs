using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;

namespace Zad1_CSharp
{
    /// <summary>
    /// This is a neuron. It has synapses.
    /// </summary>
    public class Neuron
    {
        #region variables

        protected NeuralNetwork network;

        #endregion

        #region properties

        public int ID { get; protected set; }
        public int WeightCount { get; protected set; }
        public byte[] Input { get; set; }
        public byte[] Weights { get; protected set; }
        public int Activation { get; protected set; }
        public Neuron[] OtherNeurons { get; protected set; }

        public Thread Thread { get; protected set; }

        #endregion

        #region methods

        public Neuron()
        {

        }

        ~Neuron()
        {

        }

        public void Initialize(int id, int weightCount, Neuron[] allNeurons, byte[] weights, NeuralNetwork network)
        {
            this.ID = id;
            this.WeightCount = weightCount;

            OtherNeurons = allNeurons;

            this.Weights = weights;
            this.Input = new byte[this.WeightCount];

            for (int i = 0; i < this.WeightCount; ++i)
            {
                this.Input[i] = Byte.MinValue;
            }

            this.network = network;
        }

        public void StartThread(byte[] falsePattern, AutoResetEvent ev)
        {
            falsePattern.CopyTo(this.Input, 0);

            Thread = new Thread(new ThreadStart(Run));
            Thread.Start();

            while (!Thread.IsAlive) ;
        }

        public void StopThread()
        {

        }

        public void SleepThread()
        {

        }

        public void AwakeThread()
        {

        }

        public void Shutdown()
        {

        }

        /// <summary>
        /// In this function we will:
        /// - wait until we recieve all new inputs from whole network
        /// - calculate activation function based on these inputs and send it further to other neurons
        /// - increment global recursion counter (semaphore will be used here!)
        /// - if global recursion counter exceeds its limit, awake network thread
        /// </summary>
        private void Run()
        {
            while(network.RecursionCtr < network.RecursionSteps)
            {
                System.Console.WriteLine("Neuron " + ID.ToString() + ": DUPA!");

                // increment recursion counter
                ++network.RecursionCtr;
            }
        }

        private int CalculateActivation()
        {
            return 0;
        }

        #endregion
    }
}
