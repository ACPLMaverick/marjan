using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Serialization;
using System.IO;

namespace TP_04.Model
{
    public abstract class Converter
    {
        protected string path = "RentalOffice";
        public abstract void Serialize(RentalOffice office);
        public abstract RentalOffice Deserialize();
        
    }
}
