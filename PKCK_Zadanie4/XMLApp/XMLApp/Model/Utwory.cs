using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Serialization;

namespace XMLApp.Model
{
    public class Utwory : IXmlSerializable
    {
        public Utwory()
        {
            songs = new ObservableCollection<Utwór>();
        }

        public ObservableCollection<Utwór> songs { get; set; }
        public System.Xml.Schema.XmlSchema GetSchema() { return null; }
        public void ReadXml(System.Xml.XmlReader reader)
        {
            reader.MoveToContent();
            reader.ReadStartElement();

            while (reader.NodeType != System.Xml.XmlNodeType.EndElement)
            {
                if (reader.Name == "utwór")
                {
                    Utwór song = new Utwór();
                    (song as IXmlSerializable).ReadXml(reader);
                    songs.Add(song);
                }
                reader.MoveToContent();
            }
            reader.ReadEndElement();
        }

        public void WriteXml(System.Xml.XmlWriter writer)
        {
            foreach(Utwór song in songs)
            {
                writer.WriteStartElement("utwór", "");
                (song as IXmlSerializable).WriteXml(writer);
                writer.WriteEndElement();
            }
        }
    }
}
