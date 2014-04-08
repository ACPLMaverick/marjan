import javax.swing.*;

import java.sql.Date;
import java.util.ArrayList;
import java.util.Collections;

public class Kontener implements IKontener {
	int rozmiar;
	public ArrayList<Eksponat> eksponaty;
	
	Kontener(int r)
	{
		this.rozmiar = r;
		this.eksponaty = new ArrayList<Eksponat>(rozmiar);
	}
	
	public void wyswietl()
	{
		if(eksponaty.size() != 0)
		{
			for(int i = 0; i<eksponaty.size(); i++)
			{
				System.out.println(eksponaty.get(i).toString() + "\n");
			}
		}
		else
		{
			System.out.println("Kontener jest pusty");
		}
	}
	
	public void sortuj()
	{
		String txt2 = JOptionPane.showInputDialog("Wybierze pole do posortowania (1-4)");
		int wybor = Integer.parseInt(txt2);
		switch(wybor)
		{
		case 1:
			Collections.sort(eksponaty, Eksponat.EksponatNameComparator);		//sortowanie po nazwie
			break;
		case 2:
			Collections.sort(eksponaty);		//sortowanie po numerze
			break;
		case 3:
			Collections.sort(eksponaty, Eksponat.EksponatLocComparator);		//sortowanie po lokalizacji
			break;
		case 4:
			Collections.sort(eksponaty, Eksponat.EksponatDateComparator);		//sortowanie po dacie
			break;
		default:							//brak sortowania
			break;
		}
	}
	
	public void wstaw(int miejsce)
	{
		Eksponat nowyEksponat = new Eksponat("puste", miejsce, Eksponat.lokalizacja.MAGAZYN, Date.valueOf("2014-03-18"));
		String txt = JOptionPane.showInputDialog("Jak ma nazywac sie eksponat?");
		nowyEksponat.put(txt);
		txt = JOptionPane.showInputDialog("Podaj lokalizacje [MAGAZYN, KONSERWACJA, EKSPOZYCJA, WYPOZYCZONY]: (1-4)");
		int wybor = Integer.parseInt(txt);
		switch(wybor)
		{
		case 1:
			nowyEksponat.put(Eksponat.lokalizacja.MAGAZYN);
			break;
		case 2:
			nowyEksponat.put(Eksponat.lokalizacja.KONSERWACJA);
			break;
		case 3:
			nowyEksponat.put(Eksponat.lokalizacja.EKSPOZYCJA);
			break;
		case 4:
			nowyEksponat.put(Eksponat.lokalizacja.WYPOZYCZONY);
			break;
		default:
			nowyEksponat.put(Eksponat.lokalizacja.MAGAZYN);
			break;	
		}
		txt = JOptionPane.showInputDialog("Podaj date wprowadzenia eksponatu [YYYY-MM-DD]: ");
		Date d = Date.valueOf(txt);
		nowyEksponat.put(d);
		eksponaty.add(miejsce, nowyEksponat);
	}
}
