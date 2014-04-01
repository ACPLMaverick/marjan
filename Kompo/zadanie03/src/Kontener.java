import javax.swing.*;

import java.util.ArrayList;
import java.util.Collections;

public class Kontener {
	int rozmiar;
	public ArrayList<Eksponat> eksponaty;
	
	Kontener(int r)
	{
		this.rozmiar = r;
		this.eksponaty = new ArrayList<Eksponat>(rozmiar);
	}
	
	void wyswietl()
	{
		for(int i = 0; i<eksponaty.size(); i++)
		{
			System.out.println(eksponaty.get(i).toString() + "\n");
		}
	}
	
	void sortuj()
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
}
