using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Serialization;

namespace XMLApp.Model
{
    [XmlRoot("płytoteka")]
    public class Płytoteka : IXmlSerializable
    {
        public Płytoteka() 
        { 
            info = new Info();
            gatunki = new Gatunki();
            płyty = new Płyty();
        }
        public String nagłówek { get; set; }
        public Info info { get; set; }
        public Gatunki gatunki { get; set; }
        public Płyty płyty { get; set; }

        public System.Xml.Schema.XmlSchema GetSchema() { return null; }
        public void ReadXml(System.Xml.XmlReader reader)
        {
            reader.MoveToContent();
            reader.ReadStartElement();

            nagłówek = reader.ReadElementString("nagłówek");

            if (reader.Name == "info")
            {
                (info as IXmlSerializable).ReadXml(reader);
            }

            if(reader.Name == "gatunki")
            {
                (gatunki as IXmlSerializable).ReadXml(reader);
            }

            if(reader.Name == "płyty")
            {
                (płyty as IXmlSerializable).ReadXml(reader);
            }
            reader.ReadEndElement();
        }

        public void WriteXml(System.Xml.XmlWriter writer)
        {
            writer.WriteElementString("nagłówek", nagłówek);

            writer.WriteStartElement("info");
            (info as IXmlSerializable).WriteXml(writer);
            writer.WriteEndElement();

            writer.WriteStartElement("gatunki");
            (gatunki as IXmlSerializable).WriteXml(writer);
            writer.WriteEndElement();

            writer.WriteStartElement("płyty");
            (płyty as IXmlSerializable).WriteXml(writer);
            writer.WriteEndElement();
        }
    }
}
