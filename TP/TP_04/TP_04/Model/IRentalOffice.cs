using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TP_04.Model
{
    interface IRentalOffice<T>
    {
            void put(T obj);
            Book getBook(int i);
            Reader getReader(int i);
            Rent getRent(int i);
            /*
            int count(List<T> list);
            int count(Dictionary<int, T> dict);
            int count(ObservableCollection<T> obs);
            */
            int count(string collectionType);
    }
}
