import javax.swing.*;
import java.util.Arrays;
import java.util.Comparator;
import java.sql.Date;


public class zadanie03 {

	/**
	 * @param args
	 */

	public static void wstaw(Eksponat[] e, int miejsce)
	{
		e[miejsce] = new Eksponat("puste", miejsce, Eksponat.lokalizacja.MAGAZYN, Date.valueOf("2014-03-18"));
		String txt = JOptionPane.showInputDialog("Jak ma nazywac sie eksponat?");
		e[miejsce].put(txt);
		txt = JOptionPane.showInputDialog("Gdzie ma sie znajdowac: (1-4)");
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
	}
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		String txt;
		//Date mojaData = Date.valueOf("2014-12-04");
		txt = JOptionPane.showInputDialog("Podaj rozmiar tablicy");
		//enum lokalizacja{EKSPOZYCJA, MAGAZYN, KONSERWACJA, WYPO¯YCZONY};
		int size = Integer.parseInt(txt);
		Eksponat[] anArray = new Eksponat[size];
		for(int i = 0; i<size; i++)
		{
			wstaw(anArray, i);
		}
		String txt3 = JOptionPane.showInputDialog("Podaj pierwszy indeks");
		int fromIndex = Integer.parseInt(txt3);
		String txt4 = JOptionPane.showInputDialog("Podaj drugi indeks");
		int toIndex = Integer.parseInt(txt4);
		String txt2 = JOptionPane.showInputDialog("Wybierze pole do posortowania (1-4)");
		int wybor = Integer.parseInt(txt2);
		switch(wybor)
		{
		case 1:
			Arrays.sort(anArray, fromIndex, toIndex, Eksponat.EksponatNameComparator);		//sortowanie po nazwie
			for(int j = 0; j<size; j++)
			{
				System.out.println(anArray[j]);
			}
			break;
		case 2:
			Arrays.sort(anArray, fromIndex, toIndex);		//sortowanie po numerze
			for(int j = 0; j<size; j++)
			{
				System.out.println(anArray[j]);
			}
			break;
		case 3:
			Arrays.sort(anArray, fromIndex, toIndex, Eksponat.EksponatLocComparator);
			for(int j = 0; j<size; j++)
			{
				System.out.println(anArray[j]);
			}
			break;
		}
		
		
		// TODO:
		// implementacja porównywania obiektów klasy Eksponat (zosta³o po dacie)
		// implementacja sortowania tablicy obiektów klasy Eksponat (zosta³o po dacie)
		//Eksponat mojEksponat = new Eksponat("Moj eksponat", 1, Eksponat.lokalizacja.MAGAZYN, Date.valueOf("2014-03-18"));
		//System.out.println(mojEksponat.toString());
	}
}