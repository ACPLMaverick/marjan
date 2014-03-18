import javax.swing.*;
import java.util.Arrays;
import java.sql.Date;

public class zadanie03 {

	/**
	 * @param args
	 */
	public static void wstaw(Eksponat[] e, int miejsce)
	{
		String txt = JOptionPane.showInputDialog("Podaj nazwe: ");
		e[miejsce] = new Eksponat(txt, miejsce, Eksponat.lokalizacja.MAGAZYN, Date.valueOf("2014-03-18"));
	}
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		String txt;
		txt = JOptionPane.showInputDialog("Podaj rozmiar tablicy");
		int size = Integer.parseInt(txt);
		Eksponat[] anArray = new Eksponat[size];
		for(int i = 0; i<size; i++)
		{
			wstaw(anArray, i);
		}
		
		
	}

}