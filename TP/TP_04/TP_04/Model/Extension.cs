using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TP_04.Model
{
    public static class Extension
    {
        public static List<Reader> GetMostBorrowing(this List<Reader> list)
        {
            List<Reader> mostBorrowing = new List<Reader>();
            int rentCounts = 0;
            foreach(Reader reader in list)
            {
                if (reader.rented > rentCounts) rentCounts = reader.rented;     // pozyskujemy maksymalną ilość wypożyczeń
            }
            foreach(Reader reader in list)
            {
                if (reader.rented == rentCounts) mostBorrowing.Add(reader);     // i gdy gościu będzie miał maksymalną to dodajemy go do listy
            }
            return mostBorrowing;
        }

        public static List<Reader>[] GetReadersOrderedInTens(this List<Reader> list)
        {
            int readersCount = list.Count();
            int listCount; 
            listCount = readersCount / 10 + 1;
            if (readersCount % 10 == 0) listCount--;
            List<Reader>[] specReaders = new List<Reader>[listCount];

            for (int i = 0; i < listCount; i++) specReaders[i] = new List<Reader>();

            for (int i = 0, j = 0; i < readersCount; i++)
            {
                specReaders[j].Add(list[i]);
                if (i % 10 == 0 && i != 0) j++;
            }

            return specReaders;
        }

        public static Dictionary<int, Book> GetBooksNotEverBorrowed(this Dictionary<int, Book> list)
        {
            Dictionary<int, Book> notEver = new Dictionary<int, Book>();

            foreach(KeyValuePair<int, Book> pair in list)
            {
                if (!pair.Value.wasRented) notEver.Add(pair.Key, pair.Value);
            }

            return notEver;
        }
    }
}
