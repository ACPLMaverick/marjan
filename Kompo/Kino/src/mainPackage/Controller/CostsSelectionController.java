/*
 * 
 */
package mainPackage.Controller;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Date;

import com.thoughtworks.xstream.converters.Converter;

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
public class CostsSelectionController implements SelectionController {

	private CostCollection myCollection;
	private String paramType;
	private Date paramDateMin;
	private Date paramDateMax;
	private double paramPriceMin;
	private double paramPriceMax;
	
	/**
	 * Tworzy nowy obiekt klasy CostsSelectionController.
	 *
	 * @param myRep przechowuje kolekcje kosztow.
	 */
	public CostsSelectionController(CostCollection myRep)
	{
		this.myCollection = myRep;
		paramType = "";
		paramDateMin = new Date();
		paramDateMin.setTime(- Long.MAX_VALUE);
		paramDateMax = new Date(Long.MAX_VALUE);
		paramPriceMin = - Double.MAX_VALUE;
		paramPriceMax = Double.MAX_VALUE;
	}
	
	/**
	 * Tworzy nowy obiekt klasy CostsSelectionController z konkretnymi parametrami.
	 *
	 * @param myRep przechowuje kolekcje kosztow.
	 * @param paramType okresla typ kosztow: BILET, SEANS lub LICENCJA.
	 * @param paramDateMin zadana data minimalna albo null.
	 * @param paramDateMax zadana data maksymalna albo null.
	 * @param paramPriceMin zadana cena minimalna albo 0.
	 * @param paramPriceMax zadana cena maksymalna albo 0.
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

		if(paramPriceMin == 0.0) this.paramPriceMin = - Double.MAX_VALUE;
		else this.paramPriceMin = paramPriceMin;
		
		if(paramPriceMax == 0.0) this.paramPriceMax = Double.MAX_VALUE;
		else this.paramPriceMax = paramPriceMax;
	}
	
	/**
	 * Przeksztalca kolekcje kosztow w macierz Object[][].
	 *
	 * @return Liste kosztow w postaci odpowiedniej macierzy Object[][], do wrzucenia w tabele.
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
		
		Object[][] myArray = new Object[objAL.size()][Cost.fieldsCount];
		for(int i = 0; i < objAL.size(); i++)
		{
			Object[] myObjArray = objAL.get(i);
			for(int j = 0; j < Cost.fieldsCount; j++)
			{
				myArray[i][j] = myObjArray[j];
			}
		}
		
		
		return myArray;
	}
	
	/**
	 * Przeksztalca kolekcje kosztow w dane mozliwe do umieszczenia na wykresie.
	 *
	 * @return Kolekcje w formacie odpowiednim do umieszczenia na wykresie.
	 */
	public ArrayList<ArrayList<Number>> getCollectionAsChartData()
	{
		//X - czas (long); Y - cost ca³kowity z danego dnia)
		ArrayList<ArrayList<Number>> retData = new ArrayList<ArrayList<Number>>();
		ArrayList<Number> x = new ArrayList<Number>();
		ArrayList<Number> y = new ArrayList<Number>();
		
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
				x.add(cost.getDate().getTime());
				y.add(cost.getPrice());
			}
		}
		
		ArrayList<Number> new_x = new ArrayList<Number>();
		ArrayList<Number> new_y = new ArrayList<Number>();
		
		double sum = 0;
		int i = 0;
		for(; i < x.size() - 1; i++)
		{
			if(x.get(i+1).equals(x.get(i)))
			{
				sum = sum + (Double) y.get(i);
			}
			else
			{
				new_x.add(x.get(i));
				sum = sum + (Double) y.get(i);
				new_y.add(Double.valueOf(sum));
				sum = 0;
			}
		}
		
			new_x.add(x.get(i));
			sum = sum + (Double) y.get(i);
			new_y.add(Double.valueOf(sum));
			System.out.println(String.valueOf(sum));
			sum = 0;
		
		retData.add(new_x);
		retData.add(new_y);
		
		return retData;
	}
	
	/**
	 * Przeksztalca koszt w tablice Object[].
	 *
	 * @param object Obiekt rzutowany na Cost w celu zwrocenia wartosci jego pol w postaci tablicy Object[].
	 * @return Parametry kosztu w postaci tablicy Object[].
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
		objects.add(String.format("%.2f", cost.getPrice()) + " z³");
		return objects.toArray();
	}
}