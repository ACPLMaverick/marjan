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

/**
 * 
 * Klasa s�u�y do konwersji kolekcji na macierze Object�w, co jest niezb�dne przy wrzucaniu danych do tabeli.
 * Odpowiada tak�e za sortowanie i selekcj� danych po konkretnym parametrze.
 *
 */

/*
 * REPERTUAR: wybieramy po: tytu� filmu, gatunek filmu, data filmu, 
 * KOSZTY & WYDATKI: wybieramy po: data, typ, cena
 * BILETY: wybieramy: tytu�, data, cena
 */
public class CostsSelectionController implements SelectionController {
	
	private CostCollection myCollection;
	
	private String paramType;
	private Date paramDateMin;
	private Date paramDateMax;
	private double paramPriceMin;
	private double paramPriceMax;
	
	public CostsSelectionController(CostCollection myRep)
	{
		this.myCollection = myRep;
		paramType = "";
		paramDateMin = new Date();
		paramDateMin.setTime(Long.MIN_VALUE);
		paramDateMax = new Date(Long.MAX_VALUE);
		paramPriceMin = Double.MIN_VALUE;
		paramPriceMax = Double.MAX_VALUE;
	}
	
	/**
	 * 
	 * @param myRep obiekt Repertoire
	 * @param paramTitle ��dany typ kosztu albo null
	 * @param paramDateMin ��dana data minimalna albo null
	 * @param paramDateMax ��dana data maksymalna albo null
	 * @param paramPriceMin ��dana cena minimalna albo 0
	 * @param paramPriceMax ��dana cena maksymalna albo 0
	 */
	public CostsSelectionController(CostCollection myRep, String paramType, Date paramDateMin,
										Date paramDateMax, double paramPriceMin, double paramPriceMax)
	{
		this.myCollection = myRep;
		
		if(paramType == null) this.paramType = "";
		else this.paramType = paramType;
		
		if(paramDateMin == null) 
		{
			this.paramDateMin = new Date();
			this.paramDateMin.setTime(Long.MIN_VALUE);
		}
		else this.paramDateMin = paramDateMin;
		if(paramDateMax == null)
		{
			this.paramDateMax = new Date();
			this.paramDateMax = new Date(Long.MAX_VALUE);
		}
		else this.paramDateMax = paramDateMax;

		if(paramPriceMin == 0.0) this.paramPriceMin = Double.MIN_VALUE;
		else this.paramPriceMin = paramPriceMin;
		
		if(paramPriceMax == 0.0) this.paramPriceMax = Double.MAX_VALUE;
		else this.paramPriceMax = paramPriceMax;
	}
	
	/**
	 * 
	 * @return zwraca list� koszt�w w postaci odpowiedniej macierzy Object'�w, do wrzucenia w tabel�
	 * @param type - zwraca tylko te obiekty, kt�re maj� odpowiedni typ
	 */
	
	@Override
	public Object[][] getCollectionAsObjects()
	{
		ArrayList<Object[]> objAL = new ArrayList<Object[]>();
		
		for(Cost cost : myCollection.get())
		{
			String tempParamType;
			
			if(this.paramType == "") tempParamType = cost.getType();
			else tempParamType = this.paramType;
			
			if(cost.getType().equals(tempParamType) && 
					cost.getDate().getTime() >= this.paramDateMin.getTime() &&
					cost.getDate().getTime() <= this.paramDateMax.getTime() &&
					cost.getPrice() >= this.paramPriceMin &&
					cost.getPrice() <= this.paramPriceMax)
			{
				objAL.add(getElementAsObjects(cost));
			}
		}
		
		Object[][] myArray = new Object[objAL.size()][myCollection.get().get(0).getFieldsCount()];
		for(int i = 0; i < objAL.size(); i++)
		{
			Object[] myObjArray = objAL.get(i);
			for(int j = 0; j < myCollection.get().get(i).getFieldsCount(); j++)
			{
				myArray[i][j] = myObjArray[j];
			}
		}
		
		
		return myArray;
	}
	
	
	
	/**
	 * 
	 * @return zwraca parametry kosztu w tablicy Object[],
	 * do tabeli
	 */
	@Override
	public Object[] getElementAsObjects(Object object)
	{
		Cost cost = (Cost)object;
		ArrayList<Object> objects = new ArrayList<Object>();
		
		DateFormat df = new SimpleDateFormat("dd-MM-yyyy HH:mm");
		String allDate = df.format(cost.getDate());
		String[] splitDate = allDate.split(" ");
		
		objects.add(cost.getType());
		objects.add(splitDate[0]);
		objects.add(String.format("%.2f", cost.getPrice()) + " z�");
		return objects.toArray();
	}
}