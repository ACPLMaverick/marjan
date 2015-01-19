using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Serialization;

namespace XMLApp.Model
{
    public class Gatunki : IXmlSerializable
    {
        public Gatunki()
        {
            genres = new ObservableCollection<Gatunek>();
        }

        public ObservableCollection<Gatunek> genres { get; set; }
        public System.Xml.Schema.XmlSchema GetSchema() { return null; }
        public void ReadXml(System.Xml.XmlReader reader)
        {
            reader.MoveToContent();
            reader.ReadStartElement();

            while (reader.NodeType != System.Xml.XmlNodeType.EndElement)
            {
                if (reader.Name == "gatunek")
                {
                    Gatunek g = new Gatunek();
                    (g as IXmlSerializable).ReadXml(reader);
                    genres.Add(g);
                }
                reader.MoveToContent();
            }
            reader.ReadEndElement();
        }

        public void WriteXml(System.Xml.XmlWriter writer)
        {
            foreach(Gatunek g in genres)
            {
                writer.WriteStartElement("gatunek", "");
                (g as IXmlSerializable).WriteXml(writer);
                writer.WriteEndElement();
            }
        }
    }
}
