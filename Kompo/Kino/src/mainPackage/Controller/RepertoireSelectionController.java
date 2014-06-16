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
import mainPackage.Model.Model;
import mainPackage.Model.Repertoire;
import mainPackage.Model.Seance;
import mainPackage.Model.Ticket;
import mainPackage.Model.TicketCollection;

// TODO: Auto-generated Javadoc
/**
 * 
 * Klasa sluzy do konwersji kolekcji na macierze Object[][], co jest niezbedne przy wrzucaniu danych do tabeli.
 * Odpowiada takze za sortowanie i selekcje danych po konkretnym parametrze.
 *
 */

/*
 * REPERTUAR: wybieramy po: tytu³ filmu, gatunek filmu, data filmu, 
 * KOSZTY & WYDATKI: wybieramy po: data, typ, cena
 * BILETY: wybieramy: tytu³, data, cena
 */
public class RepertoireSelectionController implements SelectionController {

	private Repertoire myCollection;
	private String paramTitle;
	private String paramGenre;
	private Date paramDateMin;
	private Date paramDateMax;
	private double paramPriceMin;
	private double paramPriceMax;
	
	/**
	 * Tworzy nowy obiekt klasy RepertoireSelectionController.
	 *
	 * @param myRep przechowuje repertuar.
	 */
	public RepertoireSelectionController(Repertoire myRep)
	{
		this.myCollection = myRep;
		paramTitle = "";
		paramGenre = "";
		paramDateMin = new Date();
		paramDateMin.setTime(- Long.MAX_VALUE);
		paramDateMax = new Date(Long.MAX_VALUE);
		paramPriceMin = - Double.MAX_VALUE;
		paramPriceMax = Double.MAX_VALUE;
	}
	
	/**
	 * Tworzy nowy obiekt klasy RepertoireSelectionController z konkretnymi parametrami.
	 *
	 * @param myRep przechowuje repertuar.
	 * @param paramTitle zadany tytul albo null.
	 * @param paramGenre zadany gatunek albo null.
	 * @param paramDateMin zadana data minimalna albo null.
	 * @param paramDateMax zadana data maksymalna albo null.
	 * @param paramPriceMin zadana cena minimalna albo 0.
	 * @param paramPriceMax zadana cena maksymalna albo 0.
	 */
	public RepertoireSelectionController(Repertoire myRep, String paramTitle, String paramGenre, Date paramDateMin,
										Date paramDateMax, double paramPriceMin, double paramPriceMax)
	{
		this.myCollection = myRep;
		
		if(paramTitle == null) this.paramTitle = "";
		else this.paramTitle = paramTitle;
		
		if(paramGenre == null) this.paramGenre = "";
		else this.paramGenre = paramGenre;
		
		if(paramDateMin == null) 
		{
			this.paramDateMin = new Date();
			this.paramDateMin.setTime(- Long.MAX_VALUE);
		}
		else this.paramDateMin = paramDateMin;
		if(paramDateMax == null)
		{
			this.paramDateMax = new Date();
			this.paramDateMax = new Date(Long.MAX_VALUE);
		}
		else this.paramDateMax = paramDateMax;

		if(paramPriceMin == 0.0) this.paramPriceMin = - Double.MAX_VALUE;
		else this.paramPriceMin = paramPriceMin;
		
		if(paramPriceMax == 0.0) this.paramPriceMax = Double.MAX_VALUE;
		else this.paramPriceMax = paramPriceMax;
	}
	
	/**
	 * Przeksztalca kolekcje seansow w macierz Object[][].
	 *
	 * @return Liste seansow w postaci odpowiedniej macierzy Object[][], do wrzucenia w tabele.
	 */

	public Object[][] getCollectionAsObjects()
	{
		ArrayList<Object[]> objAL = new ArrayList<Object[]>();
		
		for(Seance seance : myCollection.get())
		{
			String tempParamTitle, tempParamGenre;
			
			if(this.paramTitle == "") tempParamTitle = seance.getTitle();
			else tempParamTitle = this.paramTitle;
			
			if(this.paramGenre == "") tempParamGenre = seance.getGenre();
			else tempParamGenre = this.paramGenre;
			
			if(seance.getTitle().equals(tempParamTitle) && 
					seance.getGenre().equals(tempParamGenre) &&
					seance.getDate().getTime() >= this.paramDateMin.getTime() &&
					seance.getDate().getTime() <= this.paramDateMax.getTime() &&
					seance.getPrice() >= this.paramPriceMin &&
					seance.getPrice() <= this.paramPriceMax)
			{
				objAL.add(getElementAsObjects(seance));
			}
		}
		
		Object[][] myArray = new Object[objAL.size()][Seance.fieldsCount];
		for(int i = 0; i < objAL.size(); i++)
		{
			Object[] myObjArray = objAL.get(i);
			for(int j = 0; j < Seance.fieldsCount; j++)
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
	 * Przeksztalca seans w tablice Object[].
	 *
	 * @param object Obiekt rzutowany na Seance w celu zwrocenia wartosci jego pol w postaci tablicy Object[].
	 * @return Parametry seansu w postaci tablicy Object[].
	 */
	public Object[] getElementAsObjects(Object object)
	{
		Seance seance = (Seance)object;
		ArrayList<Object> objects = new ArrayList<Object>();
		
		String[] splitDate = seance.getDateAsString().split(" ");
		
		objects.add(seance.getTitle());
		objects.add(splitDate[1]);
		objects.add(splitDate[0]);
		objects.add(String.valueOf(Model.placesAvailable - seance.getSeatPlan()));
		objects.add(String.format("%.2f", seance.getPrice()) + " z³");
		return objects.toArray();
	}

}
