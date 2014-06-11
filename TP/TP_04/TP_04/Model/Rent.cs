using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization;
using System.Text;
using System.Threading.Tasks;

namespace TP_04.Model
{
    [Serializable()]
    public class Rent : ISerializable
    {
        public Book book
        {
            get;
            set;
        }
        public Reader reader
        {
            get;
            set;
        }

        public Rent()
        {
            this.book = null;
            this.reader = null;
        }

        public Rent(Book book, Reader reader)
        {
            this.book = book;
            this.reader = reader;
            this.reader.rented += 1;
            this.book.rented += 1;
            this.book.wasRented = true;
        }

        ~Rent()
        {
            if(this.reader != null) this.reader.rented -= 1;
            if(this.book != null) this.book.rented -= 1;
        }
        // overrided method ToString()
        public override String ToString()
        {
            return "\nBOOK: " + book.ToString() + "\nREADER: " + reader.ToString();
        }

        public override bool Equals(object obj)
        {
            // If parameter is null return false.
            if (obj == null)
            {
                return false;
            }

            // If parameter cannot be cast to Point return false.
            Rent secondRent = obj as Rent;
            if ((System.Object)secondRent == null)
            {
                return false;
            }

            // Return true if the fields match:
            return (this.reader == secondRent.reader) && (this.book == secondRent.book);
        }

        public static bool operator ==(Rent a, Rent b)
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
            return a.reader == b.reader && a.book == b.book;
        }

        public static bool operator !=(Rent a, Rent b) { return !(a == b); }

        public int CompareTo(object obj)
        {
            if (obj == null) return 1;

            Rent otherRent = obj as Rent;
            if (otherRent != null)
                return (this.book.CompareTo(otherRent.book));
            else
                throw new ArgumentException("Object is not a Temperature");
        }

        // do serializacji
        public Rent(SerializationInfo info, StreamingContext context)
        {
            try
            {
                this.book = (Book)info.GetValue("RentBook", typeof(Book));
                this.reader = (Reader)info.GetValue("RentReader", typeof(Reader));
            }
            catch 
            {
                Console.WriteLine("Couldn't create rent.");
            }
        }

        public virtual void GetObjectData(SerializationInfo info, StreamingContext context)
        {
            info.AddValue("RentBook", this.book, typeof(Book));
            info.AddValue("RentReader", this.reader, typeof(Reader));
        }
    }
}
