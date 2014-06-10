using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Runtime.Serialization;

namespace TP_04.Model
{
    [Serializable()]
    public class SerializableList<T> : List<T>, ISerializable
    {
        public SerializableList()
        {

        }
        public SerializableList(SerializationInfo info, StreamingContext context)
        {
            int i = 0;
            foreach (object obj in info)
            {
                this.Add((T)info.GetValue(typeof(T).Name + i.ToString(), typeof(T)));
                i++;
            }
        }

        public void GetObjectData(SerializationInfo info, StreamingContext context)
        {
            int i = 0;
            foreach(T obj in this)
            {
                info.AddValue((obj.GetType().Name + i.ToString()), obj, obj.GetType());
                i++;
            }
        }
    }
}
