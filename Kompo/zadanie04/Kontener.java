import javax.swing.*;
import java.util.ArrayList;

public class Kontener {
	int rozmiar;
	ArrayList<Eksponat> eksponaty;
	
	Kontener(int r)
	{
		this.rozmiar = r;
		this.eksponaty = new ArrayList<Eksponat>(rozmiar);
	}
}
