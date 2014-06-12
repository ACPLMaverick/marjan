package mainPackage.Model;

import java.util.ArrayList;
import java.util.Date;

/**
 * 
 * Reprezentuje ca³y repertuar, przechowuje kolekcjê seansów
 *
 */
public class Repertoire {
	
	private ArrayList<Seance> seanceList;
	
	public Repertoire()
	{
		seanceList = new ArrayList<Seance>();
	}
	
	public void add(Seance seance)
	{
		seanceList.add(seance);
	}
	
	public void delete(int i)
	{
		seanceList.remove(i);
	}
	// by index
	public Seance get(int i)
	{
		return seanceList.get(i);
	}
	
	public ArrayList<Seance> get()
	{
		return seanceList;
	}

}
