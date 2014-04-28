using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TP_01
{
    public delegate void AddedEventHandler(object sender, EventArgs e);

    class EventObservableCollection<T> : ObservableCollection<T>
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
    }
}
