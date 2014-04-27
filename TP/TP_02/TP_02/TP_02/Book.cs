using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TP_01
{
    public class Book
    {
        private string title;
        private string author;
        private int yearRelased;
        public int key;
        public bool rented
        {
            get;
            set;
        }

        public Book(string title, string author, int yearRelased, int key)
        {
            this.title = title;
            this.author = author;
            this.yearRelased = yearRelased;
            this.key = key;
            rented = false;
        }

        // overrided method ToString()
        public override String ToString()
        {
            return "TITLE: " + title + ", AUTHOR: " + author + ", YEAR RELEASE: " + yearRelased;
        }

    }
}
