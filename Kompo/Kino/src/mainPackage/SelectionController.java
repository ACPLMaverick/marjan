package mainPackage;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;

/**
 * 
 * Klasa s³u¿y do konwersji kolekcji na macierze Objectów, co jest niezbêdne przy wrzucaniu danych do tabeli.
 * Odpowiada tak¿e za sortowanie i selekcjê danych po konkretnym parametrze.
 *
 */
public class SelectionController {
	
	private Repertoire myRep;
	
	public SelectionController(Repertoire myRep)
	{
		this.myRep = myRep;
	}
	
	/**
	 * 
	 * @return zwraca repertuar w postaci odpowiedniej macierzy Object'ów, do wrzucenia w tabelê
	 */
	public Object[][] getRepertoireAsObjects()
	{
		Object[][] myArray = new Object[myRep.get().size()][myRep.get().get(0).getFieldsCount()];
		for(int i = 0; i < myRep.get().size(); i++)
		{
			Object[] myObjArray = getSeanceAsObjects(myRep.get().get(i));
			for(int j = 0; j < myRep.get().get(i).getFieldsCount(); j++)
			{
				myArray[i][j] = myObjArray[j];
			}
		}
		return myArray;
	}
	
	/**
	 * 
	 * @return zwraca parametry seansu w tablicy Object[],
	 * do tabeli
	 */
	private Object[] getSeanceAsObjects(Seance seance)
	{
		ArrayList<Object> objects = new ArrayList<Object>();
		
		DateFormat df = new SimpleDateFormat("dd-MM-yyyy HH:mm");
		String allDate = df.format(seance.getDate());
		String[] splitDate = allDate.split(" ");
		
		objects.add(seance.getTitle());
		objects.add(splitDate[1]);
		objects.add(splitDate[0]);
		objects.add(String.valueOf(Model.placesAvailable - seance.getSeatPlan()));
		objects.add(String.format("%.2f", seance.getPrice()) + " z³");
		return objects.toArray();
	}
}
