using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading.Tasks;

namespace TP_04.Model
{
    public class ConverterBin : Converter
    {
        public override void Serialize(RentalOffice office)
        {
            using (FileStream writer = File.Open(path, FileMode.Create))
            {
                BinaryFormatter formatter = new BinaryFormatter();
                try
                {
                    formatter.Serialize(writer, office);
                }
                catch (SerializationException e)
                {
                    Console.WriteLine("Failed to serialize. Reason: " + e.Message);
                }
            }

        }
        public override RentalOffice Deserialize()
        {
            RentalOffice office = new RentalOffice();
            using (FileStream reader = File.Open(path, FileMode.Open))
            {
                BinaryFormatter formatter = new BinaryFormatter();
                try
                {
                    office = (RentalOffice)formatter.Deserialize(reader);
                }
                catch (SerializationException e)
                {
                    Console.WriteLine("Failed to serialize. Reason: " + e.Message);
                }
                return office;
            }

        }
    }
}
