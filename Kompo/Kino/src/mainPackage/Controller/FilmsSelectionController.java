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

/**
 * 
 * Klasa s³u¿y do konwersji kolekcji filmów na macierze Objectów, co jest niezbêdne przy wrzucaniu danych do tabeli.
 * Ta klasa nie obs³uguje selektywnego wybierania obiektów.
 *
 */

/*
 * REPERTUAR: wybieramy po: tytu³ filmu, gatunek filmu, data filmu, 
 * KOSZTY & WYDATKI: wybieramy po: data, typ, cena
 * BILETY: wybieramy: tytu³, data, cena
 */
public class FilmsSelectionController implements SelectionController {
	
	private ArrayList<Film> myCollection;
	
	public FilmsSelectionController(ArrayList<Film> myFilms)
	{
		this.myCollection = myFilms;
	}
	
	/**
	 * 
	 * @return zwraca listê kosztów w postaci odpowiedniej macierzy Object'ów, do wrzucenia w tabelê
	 * @param type - zwraca tylko te obiekty, które maj¹ odpowiedni typ
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
	
	@Override
	public ArrayList<ArrayList<Number>> getCollectionAsChartData() {
		// TODO Auto-generated method stub
		return null;
	}
	
	/**
	 * 
	 * @return zwraca parametry seansu w tablicy Object[],
	 * do tabeli
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
