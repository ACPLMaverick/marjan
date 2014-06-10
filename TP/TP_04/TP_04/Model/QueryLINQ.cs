using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TP_04.Model
{
    public static class QueryLINQ
    {
        public static Dictionary<int, Book> GetBooksWithSpecifiedTitle(Dictionary<int, Book> list, string title)
        {
            Dictionary<int, Book> specList = new Dictionary<int, Book>();

            var specQuery =
                from pair in list
                where pair.Value.title == title
                select pair;

            foreach (var pair in specQuery)
            {
                specList.Add(pair.Key,pair.Value);
            }
            return specList;
        }

        public static Dictionary<int, Book> GetBooksWithSpecifiedIssueYear(Dictionary<int, Book> list, int minYear, int maxYear)
        {
            Dictionary<int, Book> specList = new Dictionary<int, Book>();

            var specQuery =
                from pair in list
                where ((pair.Value.yearRelased >= minYear) && (pair.Value.yearRelased <= maxYear))
                select pair;

            foreach (var pair in specQuery)
            {
                specList.Add(pair.Key, pair.Value);
            }
            return specList;
        }

        public static List<string> GetAllAuthors(Dictionary<int, Book> list)
        {
            List<string> specList = new List<string>();

            var specQuery =
                (from pair in list
                select pair.Value.author).Distinct();

            foreach(var pair in specQuery)
            {
                specList.Add(pair);
            }
            return specList;
        }

        public static void CompareLists(Dictionary<int, Book> list1, Dictionary<int, Book> list2)
        {
                var specQuery =
                   from pair1 in list1
                   from pair2 in list2
                   where (pair1.Value.CompareTo(pair2.Value) < 0)
                   select new { pair1, pair2 };
                // typy anonimowe?
                System.Console.WriteLine("Pairs where A < B: \n");
                foreach(var pair in specQuery)
                {
                    Console.WriteLine("{0} is less than {1}", pair.pair1.Value.ToString(), pair.pair2.Value.ToString());
                }
        }

        public static Book getMinElement(Dictionary<int, Book> list)
        {
            var specQuery =
                (from pair in list
                 select pair.Value.yearRelased).Max();

            foreach (var myBook in list) if (myBook.Value.yearRelased == specQuery) return myBook.Value;

            return null;
        }

        public static Reader[] getClientsWithBorrows(List<Reader> list)
        {
            int readersCount = 0;

            var specQuery =
                from reader in list
                where (reader.rented > 1)
                select reader;

            foreach (var reader in specQuery) readersCount++;
            Reader[] specReaders = new Reader[readersCount];
            int i = 0;
            foreach(var reader in specQuery)
            {
                specReaders[i] = reader;
            }
            return specReaders;
        }

        public static Dictionary<int, Rent> getDistinctBorrows(EventObservableCollection<Rent> list)
        {
            Dictionary<int, Rent> specRents = new Dictionary<int, Rent>();
            Dictionary<int, Rent> givenRents = new Dictionary<int, Rent>();

            /*
             *  List<Reserved> sortReserveds = new List<Reserved>(wypozyczenia);
            sortReserveds.Sort();
            Dictionary<int, Reserved> dictReserveds = new Dictionary<int, Reserved>();
            for (int i = 0; i < sortReserveds.Count; i++)
                dictReserveds.Add(i, sortReserveds[i]);

            var reserveds =
                from reserved in dictReserveds
                where reserved.Key == dictReserveds.Count - 1 ||
                      reserved.Value.CompareTo(dictReserveds[reserved.Key + 1]) != 0
                select reserved.Value;

            return Enumerable.ToList(reserveds);
             */
            /*
            int licznik = 0;
            foreach(Rent rent in list) 
            {
                givenRents.Add(licznik,rent);
                licznik++;
            }
             */
            for(int i = 0; i < list.Count; i++)
            {
                givenRents.Add(i, list[i]);
            }

            var specQuery =
                 (from rent in givenRents
                 where (rent.Key == givenRents.Count - 1 || (rent.Value.CompareTo(givenRents[rent.Key+1])!= 0))
                  select rent);

            int licznik = 0;
            foreach (var rent in specQuery)
            {
                specRents.Add(licznik, rent.Value);
                licznik++;
            }

            return specRents;
        }

        public static List<SimpleClass> GetBooksWithSpecifiedIssueYearList(Dictionary<int, Book> list, int minYear, int maxYear)
        {
            List<SimpleClass> simpleBooks = new List<SimpleClass>();

            var specQuery =
                from pair in list
                where ((pair.Value.yearRelased >= minYear) && (pair.Value.yearRelased <= maxYear))
                select pair;

            foreach (var pair in specQuery)
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
