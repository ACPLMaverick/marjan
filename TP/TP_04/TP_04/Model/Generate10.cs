using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TP_04.Model
{
    public class Generate10 : Generate
    {
        public Generate10(RentalOffice myOffice)
        {
            this.generatedOffice = myOffice;
        }
        public override RentalOffice generateCollection()
        {
            for (int i = 0; i < 10; i++)
            {
                generatedOffice.put(createReader());
                generatedOffice.put(createBook(i));
                generatedOffice.put(new Rent(generatedOffice.getBook(i), generatedOffice.getReader(i)));
            }
            return generatedOffice;
        }
    }
}
