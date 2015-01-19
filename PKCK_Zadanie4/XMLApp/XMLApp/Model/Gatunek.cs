using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Serialization;

namespace XMLApp.Model
{
    public class Gatunek : IXmlSerializable
    {
        public Gatunek() { }
        public String typ { get; set; }
        public String content { get; set; }
        public System.Xml.Schema.XmlSchema GetSchema() { return null; }
        public void ReadXml(System.Xml.XmlReader reader)
        {
            reader.MoveToContent();
            //--tu są elementy puste, nie wiem jak to na razie inaczej zrobić--//
            typ = reader.GetAttribute("typ");
            reader.ReadStartElement();
            content = reader.ReadElementString("nazwa", "");
            reader.ReadEndElement();
        }

        public void WriteXml(System.Xml.XmlWriter writer)
        {
            writer.WriteAttributeString("typ", typ);

            writer.WriteElementString("nazwa", content);
        }
    }
}
