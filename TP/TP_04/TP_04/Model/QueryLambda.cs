using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TP_04.Model
{
    public static class QueryLambda
    {
        public static Dictionary<int, Book> GetBooksWithSpecifiedTitle(Dictionary<int, Book> list, string title)
        {
            Dictionary<int, Book> specList = new Dictionary<int, Book>();

            var specQuery = list.Where(pair => pair.Value.title == title);

            foreach (var pair in specQuery)
            {
                specList.Add(pair.Key, pair.Value);
            }

            return specList;
        }

        public static Dictionary<int, Book> GetBooksWithSpecifiedIssueYear(Dictionary<int, Book> list, int minYear, int maxYear)
        {
            Dictionary<int, Book> specList = new Dictionary<int, Book>();

            var specQuery = list.Where(pair => (pair.Value.yearRelased >= minYear) &&
                (pair.Value.yearRelased <= maxYear));
                
            foreach (var pair in specQuery)
            {
                specList.Add(pair.Key, pair.Value);
            }
            return specList;
        }

        public static List<string> GetAllAuthors(Dictionary<int, Book> list)
        {
            List<string> specList = new List<string>();

            var specQuery = list.Select(autor => autor.Value.author).Distinct();

            foreach (var pair in specQuery)
            {
                specList.Add(pair);
            }
            return specList;
        }

        public static void CompareLists(Dictionary<int, Book> list1, Dictionary<int, Book> list2)
        {
            var specQuery =
                  from pair1 in list1
                  from pair2 in list2.Where(pair2 => pair2.Value.CompareTo(pair1.Value) > 0)
                  select new { pair1, pair2 };
            System.Console.WriteLine("Pairs where A < B: \n");
            foreach (var pair in specQuery)
            {
                Console.WriteLine("{0} is less than {1}", pair.pair1.Value.ToString(), pair.pair2.Value.ToString());
            }
        }

        public static Book getMinElement(Dictionary<int, Book> list)
        {
            var specQuery = list.Select(pair => pair.Value.yearRelased).Max();

            foreach(var myBook in list)
            {
                if (myBook.Value.yearRelased == specQuery) return myBook.Value;
            }
            return null;
        }

        public static Reader[] getClientsWithBorrows(List<Reader> list)
        {
            int readersCount = 0;

            var specQuery = list.Where(reader => reader.rented > 1);

            foreach (var reader in specQuery) readersCount++;
            Reader[] specReaders = new Reader[readersCount];
            int i = 0;
            foreach(var reader in specQuery)
            {
                specReaders[i] = reader;
            }
            return specReaders;
        }

        public static List<Rent> getDistinctBorrows(EventObservableCollection<Rent> list)
        {
             List<Rent> specRents = new List<Rent>();

             var specQuery = list.SelectMany(p => list.Where(k => !k.reader.Equals(p.reader) &&
                 !k.book.Equals(p.book))).Distinct();
                 //list.Select(rent => rent.reader).Distinct();
                 //from rent in list
                 //select new { Rent = list.Select(o => o.reader).Distinct()};

             foreach(var rent in specQuery) 
             {
                 specRents.Add(rent);
                 Console.WriteLine(rent.ToString());
             }
             return specRents;
        }

        public static List<SimpleClass> GetBooksWithSpecifiedIssueYearList(Dictionary<int, Book> list, int minYear, int maxYear)
        {
            List<SimpleClass> simpleBooks = new List<SimpleClass>();

            var specQuery = list.Where(pair => (pair.Value.yearRelased >= minYear) &&
                (pair.Value.yearRelased <= maxYear));

            foreach(var pair in specQuery)
            {
                SimpleClass mySC = new SimpleClass();
                mySC.stringValue = pair.Value.title;
                mySC.intValue = pair.Value.yearRelased;
                simpleBooks.Add(mySC);
            }
            return simpleBooks;
        }
    }
}