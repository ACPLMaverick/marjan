using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization;
using System.Text;
using System.Threading.Tasks;

namespace TP_04.Model
{
    [Serializable()]
    public class Reader : ISerializable
    {
        public string name { get; set; }
        public string secondName { get; set; }
        public double pesel { get; set; }
        public int rented
        {
            get;
            set;
        }

        public Reader()
        {
            this.name = "DEFAULT_NAME";
            this.secondName = "DEFAULT_SURNAME";
            this.pesel = -1;
            rented = 0;
        }

        public Reader(string name, string secondName, double pesel)
        {
            this.name = name;
            this.secondName = secondName;
            this.pesel = pesel;
            rented = 0;
        }

        // overrided method ToString()
        public override String ToString()
        {
            return "NAME: " + name + ", SECOND NAME: " + secondName + ", PESEL: " + pesel;
        }

        public override bool Equals(object obj)
        {
            // If parameter is null return false.
            if (obj == null)
            {
                return false;
            }

            // If parameter cannot be cast to Point return false.
            Reader secondReader = obj as Reader;
            if ((System.Object)secondReader == null)
            {
                return false;
            }

            // Return true if the fields match:
            return (this.name == secondReader.name) && (this.secondName == secondReader.secondName) && (this.pesel == secondReader.pesel);
        }

        public static bool operator ==(Reader a, Reader b)
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
            return a.name == b.name && a.secondName == b.secondName && a.pesel == b.pesel;
        }

        public static bool operator !=(Reader a, Reader b) { return !(a == b); }

        public int CompareTo(object obj)
        {
            if (obj == null) return 1;

            Reader otherReader = obj as Reader;
            if (otherReader != null)
                return this.name.CompareTo(otherReader.name);
            else
                throw new ArgumentException("Object is not a Temperature");
        }

        // do serializacji
        public Reader(SerializationInfo info, StreamingContext context)
        {
            try
            {
                this.name = info.GetString("Name");
                this.secondName = info.GetString("Surname");
                this.pesel = info.GetDouble("Pesel");
                this.rented = info.GetInt32("ReaderRented");
            }
            catch 
            {
                Console.WriteLine("Couldn't create reader.");
            }
        }

        public virtual void GetObjectData(SerializationInfo info, StreamingContext context)
        {
            info.AddValue("Name", this.name);
            info.AddValue("Surname", this.secondName);
            info.AddValue("Pesel", this.pesel);
            info.AddValue("ReaderRented", this.rented);
        }
    }

    
}
