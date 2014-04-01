import javax.swing.*;
import java.util.Arrays;
import java.util.Comparator;
import java.sql.Date;
import com.thoughtworks.xstream.XStream;
import com.thoughtworks.xstream.io.xml.xppdom.XppDom;
import org.xmlpull.v1.XmlPullParser;

public class zadanie03 {

	/**
	 * @param args
	 */

	public static void wstaw(Eksponat[] e, int miejsce)
	{
		e[miejsce] = new Eksponat("puste", miejsce, Eksponat.lokalizacja.MAGAZYN, Date.valueOf("2014-03-18"));
		String txt = JOptionPane.showInputDialog("Jak ma nazywac sie eksponat?");
		e[miejsce].put(txt);
		txt = JOptionPane.showInputDialog("Podaj lokalizacje [MAGAZYN, KONSERWACJA, EKSPOZYCJA, WYPO¯YCZONY]: (1-4)");
		int wybor = Integer.parseInt(txt);
		switch(wybor)
		{
		case 1:
			e[miejsce].put(Eksponat.lokalizacja.MAGAZYN);
			break;
		case 2:
			e[miejsce].put(Eksponat.lokalizacja.KONSERWACJA);
			break;
		case 3:
			e[miejsce].put(Eksponat.lokalizacja.EKSPOZYCJA);
			break;
		case 4:
			e[miejsce].put(Eksponat.lokalizacja.WYPO¯YCZONY);
			break;
		default:
			e[miejsce].put(Eksponat.lokalizacja.MAGAZYN);
			break;	
		}
		txt = JOptionPane.showInputDialog("Podaj date wprowadzenia eksponatu [YYYY-MM-DD]: ");
		Date d = Date.valueOf(txt);
		e[miejsce].put(d);
	}
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		Kontener kontener = new Kontener(4);
		kontener.eksponaty.add(0, new Eksponat("Marcin", 1, Eksponat.lokalizacja.MAGAZYN, Date.valueOf("2014-03-25")));
		kontener.eksponaty.add(1, new Eksponat("Janek", 5, Eksponat.lokalizacja.EKSPOZYCJA, Date.valueOf("2012-03-25")));
		kontener.eksponaty.add(2, new Eksponat("Krzysiek", 3, Eksponat.lokalizacja.KONSERWACJA, Date.valueOf("2017-03-25")));
		kontener.eksponaty.add(3, new Eksponat("Patryk", 8, Eksponat.lokalizacja.WYPO¯YCZONY, Date.valueOf("2024-03-25")));
		kontener.wyswietl();
		Serializacja ser = new Serializacja();
		ser.saveToXml(kontener.eksponaty, "D:\\ser.xml");
		Kontener kontener2 = new Kontener(1);
		kontener2.eksponaty = ser.loadFromXml("D:\\ser.xml");
		kontener.wyswietl();
	}
}