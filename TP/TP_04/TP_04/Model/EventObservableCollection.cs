using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Runtime.Serialization;

namespace TP_04.Model
{
    public delegate void AddedEventHandler(object sender, EventArgs e);
    [Serializable()]
    public class EventObservableCollection<T> : ObservableCollection<T>, ISerializable
    {
        public event AddedEventHandler Added;

        protected virtual void OnAdded(EventArgs e)
        {
            if (Added != null) Added(this, e);
        }

        protected override void InsertItem(int index, T value)
        {
            base.InsertItem(index, value);
            OnAdded(EventArgs.Empty);
        }

        public EventObservableCollection()
        {

        }
        public EventObservableCollection(SerializationInfo info, StreamingContext context)
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
