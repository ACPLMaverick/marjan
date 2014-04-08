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
	 * @throws NoALetterException 
	 */
	
	public static void main(String[] args) throws NoALetterException, NotDivisableByTwoException {
		// TODO Auto-generated method stub
		
		Kontener kontener = new Kontener(4);
		kontener.eksponaty.add(0, new Eksponat("Marcin", 1, Eksponat.lokalizacja.MAGAZYN, Date.valueOf("2014-03-25")));
		kontener.eksponaty.add(1, new Eksponat("Janek", 5, Eksponat.lokalizacja.EKSPOZYCJA, Date.valueOf("2012-03-25")));
		kontener.eksponaty.add(2, new Eksponat("Krzysiek", 3, Eksponat.lokalizacja.KONSERWACJA, Date.valueOf("2017-03-25")));
		kontener.eksponaty.add(3, new Eksponat("Patryk", 8, Eksponat.lokalizacja.WYPOZYCZONY, Date.valueOf("2024-03-25")));
		//kontener.wyswietl();
		Serializacja ser = new Serializacja(kontener.eksponaty, "ser.xml");
		ser.saveToXml();
		ser.loadFromXml();
		//ser.wyswietl();
		
		ExceptionTester.runStringTest();
		ExceptionTester.runDivisorTest();
		ExceptionTester.runALetterTest("ZENOBIUSZ");
		ExceptionTester.runDivisableByTwoTest(2309);
		
		IKontener mojIKontener = new Kontener(10);
		System.out.println(mojIKontener.statycznaStala);
		mojIKontener.wyswietl();
	}
	
	
	
}