package mainPackage;

import java.util.ArrayList;
import java.util.Date;

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
	
	// by index
	public Seance get(int i)
	{
		return seanceList.get(i);
	}
	
//	// by date
//	public ArrayList<Seance> get(Date date)
//	{
//		ArrayList<Seance> newSeances = new ArrayList<Seance>();
//		return newSeances;
//	}
//	
//	// by price
//	public ArrayList<Seance> get(Date date)
//	{
//		ArrayList<Seance> newSeances = new ArrayList<Seance>();
//		return newSeances;
//	}
//	
//	// by genre
//	public ArrayList<Seance> get(Date date)
//	{
//		ArrayList<Seance> newSeances = new ArrayList<Seance>();
//		return newSeances;
//	}
//	
//	// by name
//	public ArrayList<Seance> get(Date date)
//	{
//		ArrayList<Seance> newSeances = new ArrayList<Seance>();
//		return newSeances;
//	}
}
