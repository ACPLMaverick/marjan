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
        #region variables

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

        public bool Run(byte[] falsePattern)
        {
            System.Console.WriteLine("NeuralNetwork: Run.");
            System.Console.WriteLine("NeuralNetwork: Pattern is:       " + PatternToString(Pattern, NeuronCount));
            System.Console.WriteLine("NeuralNetwork: False Pattern is: " + PatternToString(falsePattern, NeuronCount));

            // run neurons

            // go to sleep until awaken by some neuron thread

            System.Console.WriteLine("NeuralNetwork: Finished.");

            return true;
        }

        public void Shutdown()
        {

        }

        private string PatternToString(byte[] pattern, int length)
        {
            string str = "";

            for (int i = 0; i < length; ++i )
            {
                str += pattern[i].ToString();
            }

            return str;
        }

        #endregion
    }
}
