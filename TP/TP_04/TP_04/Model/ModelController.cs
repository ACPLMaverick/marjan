using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TP_04.Model
{
    public class ModelController
    {
        private RentalOffice myOffice;
        private Generate generatorM;

        public ModelController()
        {
            generatorM = new GenerateManual(new RentalOffice());
            myOffice = generatorM.generateCollection();
        }

        public RentalOffice GetOffice()
        {
            return myOffice;
        }

        public void SerializeOffice(string path)
        {
            Converter converter = new ConverterMARJAN(path);
            converter.Serialize(myOffice);
        }

        public void DeserializeOffice(string path)
        {
            Converter converter = new ConverterMARJAN(path);
            myOffice = converter.Deserialize();
        }
    }
}
