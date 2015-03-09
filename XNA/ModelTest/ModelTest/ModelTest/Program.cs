using System;

namespace ModelTest
{
#if WINDOWS || XBOX
    static class Program
    {
        /// <summary>
        /// The main entry point for the application.
        /// </summary>
        static void Main(string[] args)
        {
            using (ModelTestGame game = new ModelTestGame())
            {
                game.Run();
            }
        }
    }
#endif
}

