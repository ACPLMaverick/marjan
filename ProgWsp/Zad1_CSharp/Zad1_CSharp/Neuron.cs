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
        public sbyte Input { get; set; }
        public sbyte[] Weights { get; protected set; }
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

        public void Initialize(int id, int weightCount, Neuron[] allNeurons, sbyte[] weights, NeuralNetwork network)
        {
            this.ID = id;
            this.WeightCount = weightCount;

            OtherNeurons = allNeurons;

            this.Weights = weights;
            this.Input = 0;

            this.network = network;
        }

        public void StartThread(sbyte[] falsePattern)
        {
            this.Input = falsePattern[ID];

            Thread = new Thread(new ThreadStart(Run));
            Thread.Start();

            while (!Thread.IsAlive) ;
        }

        public void Restart(sbyte[] weights)
        {
            this.Input = 0;
            this.Weights = weights;
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
                //System.Console.WriteLine("Neuron " + ID.ToString() + ": WSZEDŁ");
                network.SemRecursionCtr.WaitOne();

                network.Sum = 0;

                for(int i = 0; i<network.NeuronCount; i++)
                    if (i != ID)
                        network.Sum += (sbyte)(OtherNeurons[i].Input*Weights[ID*network.NeuronCount + i]);
                if (network.Sum >= 0)
                    this.Input = 1;
                else
                    this.Input = 0;

                //System.Console.WriteLine("Neuron " + ID.ToString() + ": DUPA!");
                
                // check if we can calculate activation

                network.SemRecursionCtr.Release();
                //System.Console.WriteLine("Neuron " + ID.ToString() + ": WYSZEDŁ");
                // increment recursion counter
                IncrementRecursionCtr();
            }
        }

        private int CalculateActivation()
        {
            return 0;
        }

        private void IncrementRecursionCtr()
        {
            network.SemRecursionCtr.WaitOne();

            ++network.RecursionCtr;

            network.SemRecursionCtr.Release();
        }

        #endregion
    }
}
