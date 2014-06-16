/*
 * 
 */
package mainPackage.Controller;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Date;

import mainPackage.Model.Cost;
import mainPackage.Model.CostCollection;
import mainPackage.Model.Film;
import mainPackage.Model.Model;
import mainPackage.Model.Repertoire;
import mainPackage.Model.Seance;
import mainPackage.Model.Ticket;
import mainPackage.Model.TicketCollection;

// TODO: Auto-generated Javadoc
/**
 * 
 * Klasa sluzy do konwersji kolekcji filmow na macierze Object[][], co jest niezbedne przy wrzucaniu danych do tabeli.
 * Ta klasa nie obs³uguje selektywnego wybierania obiektow.
 *
 */

/*
 * REPERTUAR: wybieramy po: tytu³ filmu, gatunek filmu, data filmu, 
 * KOSZTY & WYDATKI: wybieramy po: data, typ, cena
 * BILETY: wybieramy: tytu³, data, cena
 */
public class FilmsSelectionController implements SelectionController {

	private ArrayList<Film> myCollection;
	
	/**
	 * Tworzy nowy obiekt klasy FilmsSelectionController.
	 *
	 * @param myFilms kolekcja filmow.
	 */
	public FilmsSelectionController(ArrayList<Film> myFilms)
	{
		this.myCollection = myFilms;
	}
	
	/**
	 * Przeksztalca kolekcje filmow w macierz Object[][].
	 *
	 * @return Liste filmow w postaci odpowiedniej macierzy Object[][], do wrzucenia w tabele.
	 */

	public Object[][] getCollectionAsObjects()
	{
		ArrayList<Object[]> objAL = new ArrayList<Object[]>();
		
		for(Film film : myCollection)
		{
				objAL.add(getElementAsObjects(film));
		}
		
		Object[][] myArray = new Object[objAL.size()][Film.fieldsCount];
		for(int i = 0; i < objAL.size(); i++)
		{
			Object[] myObjArray = objAL.get(i);
			for(int j = 0; j < Film.fieldsCount; j++)
			{
				myArray[i][j] = myObjArray[j];
			}
		}
		
		
		return myArray;
	}
	
	/* (non-Javadoc)
	 * @see mainPackage.Controller.SelectionController#getCollectionAsChartData()
	 */
	@Override
	public ArrayList<ArrayList<Number>> getCollectionAsChartData() {
		// TODO Auto-generated method stub
		return null;
	}
	
	/**
	 * Przeksztalca film w tablice Object[].
	 *
	 * @param object Obiekt rzutowany na Film w celu zwrocenia wartosci jego pol w postaci tablicy Object[].
	 * @return Parametry filmu w postaci tablicy Object[].
	 */
	public Object[] getElementAsObjects(Object object)
	{
		Film film = (Film)object;
		ArrayList<Object> objects = new ArrayList<Object>();
		objects.add(film.getTitle());
		objects.add(film.getGenre());
		objects.add(String.format("%.2f", film.getPrice()) + " z³");
		objects.add(String.format("%.2f", film.getLicensePrice()) + " z³");
		return objects.toArray();
	}

}
