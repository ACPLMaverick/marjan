package mainPackage.Controller;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Date;

import mainPackage.Model.CostCollection;
import mainPackage.Model.Model;
import mainPackage.Model.Repertoire;
import mainPackage.Model.Seance;
import mainPackage.Model.TicketCollection;

/**
 * 
 * Klasa s�u�y do konwersji kolekcji na macierze Object�w, co jest niezb�dne przy wrzucaniu danych do tabeli.
 * Odpowiada tak�e za sortowanie i selekcj� danych po konkretnym parametrze.
 *
 */

/*
 * REPERTUAR: wybieramy po: tytu� filmu, gatunek filmu, data filmu, 
 * KOSZTY & WYDATKI: wybieramy po: data, typ, cena
 * BILETY: wybieramy: tytu�, data, ilo�� bilet�w (?)
 */
public class SelectionController {
	
	private Repertoire myRep;
	private CostCollection myCC;
	private TicketCollection myTC;
	
	public SelectionController(Repertoire myRep)
	{
		this.myRep = myRep;
		this.myCC = null;
		this.myTC = null;
	}
	
	public SelectionController(CostCollection myCC)
	{
		this.myCC = myCC;
		this.myRep = null;
		this.myTC = null;
	}
	
	public SelectionController(TicketCollection myTC)
	{
		this.myCC = null;
		this.myRep = null;
		this.myTC = myTC;
	}
	
	
	public Object[][] getCostsAsObjects()
	{
		return null;
	}
	
	public Object[][] getCostsAsObjects(String type)
	{
		return null;
	}
	
	public Object[][] getCostsAsObjects(double priceMin, double priceMax)
	{
		return null;
	}
	
	public Object[][] getTicketsAsObjects()
	{
		return null;
	}
	
	public Object[][] getTicketsAsObjects(String title)
	{
		return null;
	}
	
	public Object[][] getTicketsAsObjects(Date dateMin, Date dateMax)
	{
		return null;
	}
	/**
	 * 
	 * @return zwraca repertuar w postaci odpowiedniej macierzy Object'�w, do wrzucenia w tabel�
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
	 * @return zwraca repertuar w postaci odpowiedniej macierzy Object'�w, do wrzucenia w tabel�
	 * @param title - zwraca tylko te obiekty, kt�re maj� odpowiedni tytu� b�d� gatunek
	 */
	public Object[][] getRepertoireAsObjects(String title)
	{
		ArrayList<Object[]> objAL = new ArrayList<Object[]>();
		
		for(Seance seance : myRep.get())
		{
			if(seance.getTitle().equals(title) || seance.getGenre().equals(title))
			{
				objAL.add(getSeanceAsObjects(seance));
			}
		}
		
		Object[][] myArray = new Object[objAL.size()][myRep.get().get(0).getFieldsCount()];
		for(int i = 0; i < objAL.size(); i++)
		{
			Object[] myObjArray = objAL.get(i);
			for(int j = 0; j < myRep.get().get(i).getFieldsCount(); j++)
			{
				myArray[i][j] = myObjArray[j];
			}
		}
		
		
		return myArray;
	}
	
	/**
	 * 
	 * @return zwraca repertuar w postaci odpowiedniej macierzy Object'�w, do wrzucenia w tabel�
	 * @param title - zwraca tylko te obiekty, kt�re maj� odpowiedni tytu� b�d� gatunek
	 */
	public Object[][] getRepertoireAsObjects(Date date)
	{
		ArrayList<Object[]> objAL = new ArrayList<Object[]>();
		
		for(Seance seance : myRep.get())
		{
			if(seance.getDate().equals(date))
			{
				objAL.add(getSeanceAsObjects(seance));
			}
		}
		
		Object[][] myArray = new Object[objAL.size()][myRep.get().get(0).getFieldsCount()];
		for(int i = 0; i < objAL.size(); i++)
		{
			Object[] myObjArray = objAL.get(i);
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
		objects.add(String.format("%.2f", seance.getPrice()) + " z�");
		return objects.toArray();
	}
}
