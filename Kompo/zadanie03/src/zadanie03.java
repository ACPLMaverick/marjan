import javax.swing.*;
import java.util.Arrays;
import java.sql.Date;


public class zadanie03 {

	/**
	 * @param args
	 */
	public static void wstaw(Eksponat[] e, int miejsce)
	{
		//String txt = JOptionPane.showInputDialog("Podaj nazwe: ");
		e[miejsce] = new Eksponat("nowy eksponat", miejsce, Eksponat.lokalizacja.MAGAZYN, Date.valueOf("2014-03-18"));
	}
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		String txt;
		//Date mojaData = Date.valueOf("2014-12-04");
		txt = JOptionPane.showInputDialog("Podaj rozmiar tablicy");
		//enum lokalizacja{EKSPOZYCJA, MAGAZYN, KONSERWACJA, WYPO¯YCZONY};
		//enum lokalizacja{EKSPOZYCJA, MAGAZYN, KONSERWACJA, WYPO¯YCZONY};String txt;
		int size = Integer.parseInt(txt);
		Eksponat[] anArray = new Eksponat[size];
		for(int i = 0; i<size; i++)
		{
			wstaw(anArray, i);
		}
		
		// TODO:
		// implementacja porównywania obiektów klasy Eksponat
		// implementacja sortowania tablicy obiektów klasy Eksponat
		Eksponat mojEksponat = new Eksponat("Moj eksponat", 1, Eksponat.lokalizacja.MAGAZYN, Date.valueOf("2014-03-18"));
		System.out.println(mojEksponat.toString());
	}

}