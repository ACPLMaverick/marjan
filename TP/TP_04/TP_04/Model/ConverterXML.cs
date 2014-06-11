using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Serialization;
using System.IO;

namespace TP_04.Model
{
    public class ConverterXML : Converter
    {
        protected new string path;

        public ConverterXML()
        {
            this.path = base.path + ".xml";
        }

        public ConverterXML(string path)
        {
            this.path = path;
        }

        public override void Serialize(RentalOffice office)
        {
            XmlSerializer serializer = new XmlSerializer(office.GetType());
            FileStream writer = new FileStream(path, FileMode.Create);
            serializer.Serialize(writer, office);
            writer.Close();
        }

        public override RentalOffice Deserialize()
        {
            RentalOffice office = new RentalOffice();
            List<Book> tempBookList = new List<Book>();
            Dictionary<int, Book> tempBookDictionary = new Dictionary<int, Book>();
            XmlSerializer deserializer = new XmlSerializer(office.GetType());
            FileStream reader = new FileStream(path, FileMode.Open);
            office = (RentalOffice)deserializer.Deserialize(reader);
            reader.Close();
            return office;
        }
    }
}
