using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Zad2
{
    public class UserFixed : User
    {
        #region constants

        private const int ADD_VALUE = 1;

        #endregion

        #region variables

        private int pos;

        #endregion

        #region properties

        #endregion

        #region methods

        public UserFixed(uint id, Database db, int position) : base(id, ADD_VALUE, db)
        {
            pos = position;
            Type = UserType.FIXED;
            typeStr = "F";
        }

        protected override int SelectValue()
        {
            return pos;
        }

        #endregion
    }
}
