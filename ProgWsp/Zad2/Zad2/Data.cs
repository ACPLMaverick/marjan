using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Zad2
{
    public class Data
    {
        public int Value { get; set; }
        public User.UserType LastUserAccess { get; set; }

        public Data()
        {
            Value = 0;
            LastUserAccess = User.UserType.NONE;
        }
    }
}
