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

        #endregion

        #region variables

        private Random random;

        #endregion

        #region properties

        #endregion

        #region methods

        public UserRandom(uint id, int addValue, Database db) : base(id, addValue, db)
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
