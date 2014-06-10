using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization;
using System.Text;
using System.Threading.Tasks;

namespace TP_04.Model
{
	[Serializable()]
    public class Book : IComparable, ISerializable
    {
        public string title { get; set; }
        public string author { get; set; }
        public int yearRelased { get; set; }
        public int key { get; set; }
        public int rented
        {
            get;
            set;
        }
        public bool wasRented;

        public Book()
        {
            this.title = "DEFAULT_TITLE";
            this.author = "DEFAULT_AUTHOR";
            this.yearRelased = -1;
            this.key = -1;
            rented = 0;
        }

        public Book(string title, string author, int yearRelased, int key)
        {
            this.title = title;
            this.author = author;
            this.yearRelased = yearRelased;
            this.key = key;
            rented = 0;
        }

        // overrided method ToString()
        public override String ToString()
        {
            return "TITLE: " + title + ", AUTHOR: " + author + ", YEAR RELEASE: " + yearRelased;
        }

        public int CompareTo(object obj)
        {
            if (obj is Book)
            {
                Book secondBook = (Book)obj;
                if (this.author == secondBook.author) return this.title.CompareTo(secondBook.title);
                else return this.author.CompareTo(secondBook.author);
            }
            else throw new ArgumentException("Argument is not of a Book type");
        }

        public override bool Equals(object obj)
        {
            // If parameter is null return false.
            if (obj == null)
            {
                return false;
            }

            // If parameter cannot be cast to Point return false.
            Book secondBook = obj as Book;
            if ((System.Object)secondBook == null)
            {
                return false;
            }

            // Return true if the fields match:
            return (this.author == secondBook.author) && (this.title == secondBook.title);
        }

        public static bool operator ==(Book a, Book b)
        {
            // If both are null, or both are same instance, return true.
            if (System.Object.ReferenceEquals(a, b))
            {
                return true;
            }

            // If one is null, but not both, return false.
            if (((object)a == null) || ((object)b == null))
            {
                return false;
            }

            // Return true if the fields match:
            return a.author == b.author && a.title == b.title;
        }

        public static bool operator !=(Book a, Book b) { return !(a == b); }

        // do serializacji
        public Book(SerializationInfo info, StreamingContext context)
        {
            try
            {
                this.title = info.GetString("Title");
                this.author = info.GetString("Author");
                this.yearRelased = info.GetInt32("YearRelased");
                this.key = info.GetInt32("Key");
                this.rented = info.GetInt32("BookRented");
                this.wasRented = info.GetBoolean("WasRented");
            }
            catch 
            {
                Console.WriteLine("Couldn't create book.");
            }
        }

        public virtual void GetObjectData(SerializationInfo info, StreamingContext context)
        {
            info.AddValue("Title", this.title);
            info.AddValue("Author", this.author);
            info.AddValue("YearRelased", this.yearRelased);
            info.AddValue("Key", this.key);
            info.AddValue("BookRented", this.rented);
            info.AddValue("WasRented", this.wasRented);
        }
    }
}
