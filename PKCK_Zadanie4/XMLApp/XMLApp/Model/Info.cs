using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml;
using System.Xml.Serialization;

namespace XMLApp.Model
{
    public class Info : IXmlSerializable
    {
        public Info() 
        {

        }
        public String Imię_i_nazwisko1 { get; set; }
        public int Nr_albumu1 { get; set; }
        public String Imię_i_nazwisko2 { get; set; }
        public int Nr_albumu2 { get; set; }
        public String laboratorium { get; set; }
        public String prowadząca { get; set; }
        public System.Xml.Schema.XmlSchema GetSchema() { return null; }
        public void ReadXml(System.Xml.XmlReader reader)
        {
            reader.MoveToContent();
            reader.ReadStartElement();

            Imię_i_nazwisko1 = reader.ReadElementString("imię_i_nazwisko", "");
            Nr_albumu1 = reader.ReadElementContentAsInt("nr_albumu", "");
            Imię_i_nazwisko2 = reader.ReadElementString("imię_i_nazwisko", "");
            Nr_albumu2 = reader.ReadElementContentAsInt("nr_albumu", "");

            laboratorium = reader.ReadElementString("laboratorium", "");
            prowadząca = reader.ReadElementString("prowadząca", "");
            reader.ReadEndElement();
        }

        public void WriteXml(System.Xml.XmlWriter writer)
        {
            writer.WriteElementString("imię_i_nazwisko", Imię_i_nazwisko1);
            writer.WriteElementString("nr_albumu", Nr_albumu1.ToString());
            writer.WriteElementString("imię_i_nazwisko", Imię_i_nazwisko2);
            writer.WriteElementString("nr_albumu", Nr_albumu2.ToString());

            writer.WriteElementString("laboratorium", laboratorium);
            writer.WriteElementString("prowadząca", prowadząca);
        }

    }
}
