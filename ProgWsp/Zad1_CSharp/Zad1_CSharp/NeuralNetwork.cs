using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
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

        #endregion

        #region properties

        public int NeuronCount { get; private set; }
        public int RecursionSteps { get; private set; }
        public sbyte[] Pattern { get; private set; } // Pattern length = NeuronCount. Always.

        public Neuron[] Neurons { get; private set; }

        public Semaphore SemRecursionCtr { get; private set; }
        public int RecursionCtr { get; set; }
        public sbyte Sum { get; set; }

        #endregion

        #region methods

        public void Initialize(int neuronCount, int recursionSteps, sbyte[] pattern)
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

            sbyte[] tab = CreateWeightMatrix();
            DrawWeightMatrix(tab);

            for (int i = 0; i < this.NeuronCount; ++i)
            {
                this.Neurons[i].Initialize(i, NeuronCount, this.Neurons, tab, this);
            }

            this.SemRecursionCtr = new Semaphore(1, 1, "Przemek");
        }

        public sbyte[] Run(sbyte[] falsePattern)
        {
            // run neurons
            for (int i = 0; i < NeuronCount; ++i )
            {
                Neurons[i].StartThread(falsePattern);
            }

            // go to sleep until all neuron threads finish
            for (int i = 0; i < NeuronCount; ++i)
            {
                Neurons[i].Thread.Join(TIMEOUT);
            }


            if(RecursionCtr >= RecursionSteps)
            {
                System.Console.WriteLine("NeuralNetwork: Finished.");

                sbyte[] ret = new sbyte[NeuronCount];
                for (int i = 0; i < NeuronCount; ++i )
                {
                    ret[i] = (sbyte)(Neurons[i].Input);
                }

                return ret;
            }
            else
            {
                // stop all threads as they have stalled somehow
                for (int i = 0; i < NeuronCount; ++i)
                {
                    Neurons[i].Thread.Abort();
                }

                System.Console.WriteLine("NeuralNetwork: Failed. Reason: Timeout.");
                return null;
            }
        }

        public void Shutdown()
        {

        }


        public sbyte[] CreateWeightMatrix()
        {
            sbyte[] ret = new sbyte[NeuronCount * NeuronCount];
            sbyte sum;

            for (int j = 0; j < NeuronCount; j++)
            {
                for (int i = j; i < NeuronCount; i++)
                {
                    if (i == j)
                        ret[j*NeuronCount + i] = 0;
                    else
                    {
                        sum = 0;
                        sum += (sbyte) ((Pattern[i]*2 - 1)*(Pattern[j]*2 - 1));
                        ret[j*NeuronCount + i] = sum;
                        ret[i*NeuronCount + j] = sum;
                    }
                }
            }

            return ret;
        }

        public void DrawWeightMatrix(sbyte[] matrix)
        {
            for (int i = 0; i < NeuronCount; i++)
            {
                for (int j = 0; j < NeuronCount; j++)
                {
                    Console.Write(matrix[i * NeuronCount + j]);
                }
                Console.WriteLine("");
            }
        }


        #endregion
    }
}
