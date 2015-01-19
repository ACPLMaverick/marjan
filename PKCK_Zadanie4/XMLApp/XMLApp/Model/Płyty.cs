using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Serialization;

namespace XMLApp.Model
{
    public class Płyty : IXmlSerializable
    {
        public Płyty() 
        {
            płyty = new ObservableCollection<Płyta>();
        }
        public ObservableCollection<Płyta> płyty { get; set; }
        public System.Xml.Schema.XmlSchema GetSchema() { return null; }
        public void ReadXml(System.Xml.XmlReader reader)
        {
            reader.MoveToContent();
            reader.ReadStartElement();

            //--deserializacja listy--//
            while (reader.NodeType != System.Xml.XmlNodeType.EndElement)
            {
                if (reader.Name == "płyta")
                {
                    Płyta cd = new Płyta();
                    (cd as IXmlSerializable).ReadXml(reader);
                    płyty.Add(cd);
                }
                reader.MoveToContent();
            }
 
            reader.ReadEndElement();
        }
        public void WriteXml(System.Xml.XmlWriter writer)
        {
            foreach(Płyta cd in płyty)
            {
                writer.WriteStartElement("płyta", "");
                (cd as IXmlSerializable).WriteXml(writer);
                writer.WriteEndElement();
            }
        }
    }
}
