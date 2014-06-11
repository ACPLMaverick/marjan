package mainPackage;

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
	
	/**
	 * 
	 * @return zwraca repertuar w postaci odpowiedniej macierzy Object'ów, do wrzucenia w tabelê
	 */
	public Object[][] getAsObjectMatrix()
	{
		Object[][] myArray = new Object[seanceList.size()][seanceList.get(0).getFieldsCount()];
		for(int i = 0; i < seanceList.size(); i++)
		{
			Object[] myObjArray = seanceList.get(i).getParamsAsObjectArray();
			for(int j = 0; j < seanceList.get(i).getFieldsCount(); j++)
			{
				myArray[i][j] = myObjArray[j];
			}
		}
		return myArray;
	}

}
