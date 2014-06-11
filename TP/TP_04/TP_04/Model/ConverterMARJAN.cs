using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Runtime.Serialization;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;
using System.Reflection;

namespace TP_04.Model
{
    public class ConverterMARJAN : Converter
    {
        protected new string path;

        public ConverterMARJAN()
        {
            this.path = base.path + ".marjan";
        }

        public ConverterMARJAN(string path)
        {
            this.path = path;
        }

        public override void Serialize(RentalOffice office)
        {
            //foreach(Rent reader in office.getRentCollection()) WriteObject(reader, typeof(Rent));
            StreamWriter writer = new StreamWriter(path);
            WriteObject(office, typeof(RentalOffice), writer);
            writer.Close();
        }

        public override RentalOffice Deserialize()
        {
            StreamReader reader = new StreamReader(path);
            RentalOffice myOffice = (RentalOffice)ReadObject(typeof(RentalOffice),  reader);
            
            //string file = reader.ReadToEnd();
            //Console.Write(file);
            reader.Close();
            return myOffice;
        }

        private object ReadObject(Type type, StreamReader reader)
        {
            SerializationInfo myInfo = new SerializationInfo(type, new FormatterConverter());

            string nameBuffer = "";
            string valueBuffer = "";
            char currentChar;

            if (type == typeof(int))
            {
                valueBuffer = ReadNext(reader);
                Type myCurrentType = GetNextType(nameBuffer);
                return Convert.ToInt32(valueBuffer);
            }
            else if (type == typeof(double))
            {
                valueBuffer = ReadNext(reader);
                Type myCurrentType = GetNextType(nameBuffer);
                return Convert.ToDouble(valueBuffer);
            }
            else if (type == typeof(bool))
            {
                valueBuffer = ReadNext(reader);
                Type myCurrentType = GetNextType(nameBuffer);
                return Convert.ToBoolean(valueBuffer);
            }
            else if (type == typeof(string))
            {
                valueBuffer = ReadNext(reader);
                Type myCurrentType = GetNextType(nameBuffer);
                return valueBuffer;
            }
            //else throw new Exception("Bad type recognized."); 

            int i = 0;
            while(!reader.EndOfStream)
            {
                currentChar = Convert.ToChar(reader.Read());
                if (currentChar == '<')
                {
                    nameBuffer = ReadNext(reader);
                }
                if (reader.Peek() == '=')
                {
                    currentChar = Convert.ToChar(reader.Read());
                    if (type == typeof(SerializableList<Reader>) || type == typeof(SerializableList<Book>) || type == typeof(EventObservableCollection<Rent>))
                    {
                        myInfo.AddValue(nameBuffer + i.ToString(), ReadObject(GetNextType(nameBuffer), reader), GetNextType(nameBuffer));
                        i++;
                    }
                    else
                    {
                        myInfo.AddValue(nameBuffer, ReadObject(GetNextType(nameBuffer), reader), GetNextType(nameBuffer));
                    }
                }
                if (currentChar == '>')
                {
                    if(reader.Peek() == '>') break;
                }
            }

            Type[] neededTypes = new Type[] { typeof(SerializationInfo), typeof(StreamingContext) };
            object[] neededObjects = new object[] { myInfo, new StreamingContext() };
            //Type[] neededTypes = new Type[] {  };
            //object[] neededObjects = new object[] {  };
            ConstructorInfo myCtor = type.GetConstructor(neededTypes);
            object myObject = null;
            try
            {
                myObject = myCtor.Invoke(neededObjects);
                //Console.WriteLine("Created for " + type.Name + " " + nameBuffer + " " + valueBuffer);
            }
            catch
            {
                Console.WriteLine("Nullreferenceexception for " + type.Name + " " + nameBuffer + " " + valueBuffer);
            }
            
            return myObject;
        }

        private string ReadNext(StreamReader reader)
        {
            string next = "";
            do
            {
                next = next + Convert.ToChar(reader.Read());
            } while (reader.Peek() != '>' && reader.Peek() != '<' && reader.Peek() != '=');

            if (next.Contains("Reader") && next.Length < 8) next = "Reader";
            else if (next.Contains("Book") && next.Length < 6) next = "Book";
            else if (next.Contains("Rent") && next.Length < 6) next = "Rent";
            //Console.WriteLine(next);
            return next;
        }

        private Type GetNextType(string name)
        {
            Type myType;
            if (name == "readers") myType = typeof(SerializableList<Reader>);
            else if (name == "books") myType = typeof(SerializableList<Book>);
            else if (name == "rents") myType = typeof(EventObservableCollection<Rent>);
            else if (name.Contains("Reader") && name != "RentReader" && name != "ReaderRented") myType = typeof(Reader);
            else if (name == "RentReader") myType = typeof(Reader);
            else if (name.Contains("Book") && name != "RentBook" && name != "BookRented") myType = typeof(Book);
            else if (name == "RentBook") myType = typeof(Book);
            else if (name.Contains("Rent") && name != "ReaderRented" && name != "BookRented" && name != "WasRented") myType = typeof(Rent);
            else if (name == "ReaderRented") myType = typeof(int);
            else if (name == "BookRented") myType = typeof(int);
            else if (name == "YearRelased") myType = typeof(int);
            else if (name == "Key") myType = typeof(int);
            else if (name == "Pesel") myType = typeof(double);
            else if (name == "WasRented") myType = typeof(bool);
            else myType = typeof(string);
            return myType;
        }


        private void WriteObject(ISerializable myObj, Type type, StreamWriter writer)
        {
            SerializationInfo myInfo = new SerializationInfo(type, new FormatterConverter());
            myObj.GetObjectData(myInfo, new StreamingContext());

            //writer.Write("<" + type.Name + ">");
            foreach(var obj in myInfo)
            {
                writer.Write("<" + obj.Name + "=");
                if (obj.Value is ISerializable)
                {
                    WriteObject((ISerializable)obj.Value, obj.Value.GetType(), writer);
                }
                else
                {
                    writer.Write(obj.Value);
                }
                writer.Write(">");
            }
            //writer.Write("</" + type.Name + ">");
            writer.Flush();
        }
    }
}
