using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Zad2
{
    public class UserRandom : User
    {
        #region constants

        private const int ADD_VALUE = -1;

        #endregion

        #region variables

        private Random random;

        #endregion

        #region properties

        #endregion

        #region methods

        public UserRandom(uint id, Database db) : base(id, ADD_VALUE, db)
        {
            random = new Random();
            Type = UserType.RANDOM;
            typeStr = "R";
        }

        protected override int SelectValue()
        {
            int val = random.Next((int)db.ItemCount);

            return val;
        }

        #endregion
    }
}
