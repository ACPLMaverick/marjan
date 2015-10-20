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

        #endregion

        #region variables

        private int pos;

        #endregion

        #region properties

        #endregion

        #region methods

        public UserFixed(uint id, int addValue, Database db, int position) : base(id, addValue, db)
        {
            pos = position;
        }

        protected override int SelectValue()
        {
            return pos;
        }

        #endregion
    }
}
