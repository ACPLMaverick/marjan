using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Zad1_CSharp
{
    /// <summary>
    /// This class is a base for a neural network. It governs every neuron's activity and their synchronization, using a binary semaphore.
    /// </summary>
    public class NeuralNetwork
    {
        #region variables

        #endregion

        #region properties

        public int NeuronCount { get; private set; }
        public int RecursionSteps { get; private set; }
        public int Pattern { get; private set; }
        public Neuron[] Neurons { get; private set; }

        #endregion

        #region methods

        public void Initialize(int neuronCount, int recursionSteps, int pattern)
        {
            this.NeuronCount = neuronCount;
            this.RecursionSteps = recursionSteps;
            this.Pattern = pattern;

            this.Neurons = new Neuron[this.NeuronCount];

        }

        public bool Run(int falsePattern)
        {
            System.Console.WriteLine("NeuralNetwork: Dummy run.");
            return true;
        }

        public void Shutdown()
        {

        }

        #endregion
    }
}
