using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Serialization;

namespace XMLApp.Model
{
    public class Płyta : IXmlSerializable
    {
        public Płyta() 
        {
            gatunekPłyty = new Gatunek();
            utwory = new Utwory();
        }

        System.Globalization.CultureInfo enUS = new System.Globalization.CultureInfo("en-US");
        public String ID { get; set; }
        public String tytuł { get; set; }
        public Gatunek gatunekPłyty { get; set; }
        public String autor { get; set; }
        public String kraj { get; set; }
        private DateTime year { get; set; }
        public String rok { get; set; }
        public DateTime czas_całkowity { get; set; }
        public String waluta { get; set; }

        public Decimal cena { get; set; }
        public Utwory utwory { get; set; }
        public System.Xml.Schema.XmlSchema GetSchema() { return null; }
        public void ReadXml(System.Xml.XmlReader reader)
        {
            reader.MoveToContent();
            ID = reader.GetAttribute("ID", "");
            reader.ReadStartElement();

            gatunekPłyty.typ = reader.GetAttribute("gatunekPłyty", "");
            if (reader.Name == "gatunekPłyty")
            {
                (gatunekPłyty as IXmlSerializable).ReadXml(reader);
            }
            tytuł = reader.ReadElementString("tytuł", "");
            kraj = reader.GetAttribute("kraj");
            autor = reader.ReadElementString("autor", "");
            year = DateTime.ParseExact(reader.ReadElementString("rok",""), "yyyy", enUS);
            rok = year.Year.ToString();
            czas_całkowity = DateTime.ParseExact(reader.ReadElementString("czas_całkowity", ""), "hh:mm:ss", enUS);
            waluta = reader.GetAttribute("waluta");
            cena = reader.ReadElementContentAsDecimal("cena", "");
            if(reader.Name == "utwory")
            {
                (utwory as IXmlSerializable).ReadXml(reader);
            }

            reader.ReadEndElement();
        }
        public void WriteXml(System.Xml.XmlWriter writer)
        {
            writer.WriteAttributeString("ID", ID);

            writer.WriteStartElement("tytuł", "");
            writer.WriteAttributeString("gatunekPłyty", gatunekPłyty.typ);
            writer.WriteString(tytuł);
            writer.WriteEndElement();

            writer.WriteStartElement("autor", "");
            writer.WriteAttributeString("kraj", kraj);
            writer.WriteString(autor);
            writer.WriteEndElement();

            year = DateTime.ParseExact(rok, "yyyy", enUS);
            writer.WriteElementString("rok", year.ToString("yyyy"));
            writer.WriteElementString("czas_całkowity", czas_całkowity.ToString("HH:mm:ss"));

            writer.WriteStartElement("cena", "");
            writer.WriteAttributeString("waluta", waluta);
            writer.WriteString(cena.ToString(enUS));
            writer.WriteEndElement();

            writer.WriteStartElement("utwory");
            (utwory as IXmlSerializable).WriteXml(writer);
            writer.WriteEndElement();
        }
    }
}
