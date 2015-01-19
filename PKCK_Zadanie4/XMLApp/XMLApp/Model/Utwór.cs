using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Serialization;

namespace XMLApp.Model
{
    public class Utwór : IXmlSerializable
    {
        public Utwór() { }
        System.Globalization.CultureInfo plPL = new System.Globalization.CultureInfo("pl-PL"); 
        public String tytuł { get; set; }
        public String numer { get; set; }
        public DateTime czas { get; set; }
        public System.Xml.Schema.XmlSchema GetSchema() { return null; }
        public void ReadXml(System.Xml.XmlReader reader)
        {
            reader.MoveToContent();
            numer = reader.GetAttribute("numer", "");
            reader.ReadStartElement();

            tytuł = reader.ReadElementString("tytuł", "");
            czas = DateTime.ParseExact(reader.ReadElementString("czas",""), "hh:mm:ss", plPL);

            reader.ReadEndElement();
        }
        public void WriteXml(System.Xml.XmlWriter writer)
        {
            writer.WriteAttributeString("numer", numer);
            writer.WriteElementString("tytuł", tytuł);
            writer.WriteElementString("czas", czas.ToString("HH:mm:ss"));
        }
    }
}
