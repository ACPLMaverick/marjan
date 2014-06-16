/*
 * 
 */
package mainPackage.Controller;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;

import mainPackage.Model.Ticket;
import mainPackage.Model.TicketCollection;

// TODO: Auto-generated Javadoc
/**
 * Klasa sluzy do konwersji kolekcji na macierze Object[][], co jest niezbedne przy wrzucaniu danych do tabeli.
 * Odpowiada takze za sortowanie i selekcje danych po konkretnym parametrze.
 * 
 */
public class TicketsSelectionController implements SelectionController {

	private TicketCollection myCollection;
	private String paramTitle;
	private Date paramDateMin;
	private Date paramDateMax;
	private double paramPriceMin;
	private double paramPriceMax;
	
	/**
	 * Tworzy nowy obiekt klasy TicketsSelectionController.
	 *
	 * @param myRep przechowuje kolekcje biletow.
	 */
	public TicketsSelectionController(TicketCollection myRep)
	{
		this.myCollection = myRep;
		paramTitle = "";
		paramDateMin = new Date();
		paramDateMin.setTime(Long.MIN_VALUE);
		paramDateMax = new Date(Long.MAX_VALUE);
		paramPriceMin = Double.MIN_VALUE;
		paramPriceMax = Double.MAX_VALUE;
	}
	
	/**
	 * Tworzy nowy obiekt klasy TicketsSelectionController z konkretnymi parametrami.
	 *
	 * @param myRep przechowuje kolekcje biletow.
	 * @param paramTitle zadany typ kosztu albo null.
	 * @param paramDateMin zadana data minimalna albo null.
	 * @param paramDateMax zadana data maksymalna albo null.
	 * @param paramPriceMin zadana cena minimalna albo 0.
	 * @param paramPriceMax zadana cena maksymalna albo 0.
	 */
	public TicketsSelectionController(TicketCollection myRep, String paramTitle, Date paramDateMin,
										Date paramDateMax, double paramPriceMin, double paramPriceMax)
	{
		this.myCollection = myRep;
		
		if(paramTitle == null) this.paramTitle = "";
		else this.paramTitle = paramTitle;
		
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
	 * Przeksztalca kolekcje biletow w macierz Object[][].
	 *
	 * @return Liste biletow w postaci odpowiedniej macierzy Object[][], do wrzucenia w tabele.
	 */
	
	@Override
	public Object[][] getCollectionAsObjects()
	{
		ArrayList<Object[]> objAL = new ArrayList<Object[]>();
		
		for(Ticket ticket : myCollection.get())
		{
			String tempParamTitle;
			
			if(this.paramTitle == "") tempParamTitle = ticket.getSeance().getTitle();
			else tempParamTitle = this.paramTitle;
			
			if(ticket.getSeance().getTitle().equals(tempParamTitle) && 
					ticket.getSeance().getDate().getTime() >= this.paramDateMin.getTime() &&
					ticket.getSeance().getDate().getTime() <= this.paramDateMax.getTime() &&
					ticket.getPrice() >= this.paramPriceMin &&
					ticket.getPrice() <= this.paramPriceMax)
			{
				objAL.add(getElementAsObjects(ticket));
			}
		}
		
		Object[][] myArray = new Object[objAL.size()][Ticket.fieldsCount];
		for(int i = 0; i < objAL.size(); i++)
		{
			Object[] myObjArray = objAL.get(i);
			for(int j = 0; j < Ticket.fieldsCount; j++)
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
	 * Przeksztalca bilet w tablice Object[].
	 *
	 * @param object Obiekt rzutowany na Ticket w celu zwrocenia wartosci jego pol w postaci tablicy Object[].
	 * @return Parametry seansu w postaci tablicy Object[].
	 */
	@Override
	public Object[] getElementAsObjects(Object object)
	{
		Ticket ticket = (Ticket)object;
		ArrayList<Object> objects = new ArrayList<Object>();
		
		DateFormat df = new SimpleDateFormat("dd-MM-yyyy HH:mm");
		String allDate = df.format(ticket.getSeance().getDate());
		String[] splitDate = allDate.split(" ");
		
		objects.add(ticket.getSeance().getTitle());
		objects.add(splitDate[0]);
		objects.add(String.format("%.2f", ticket.getPrice()) + " z³");
		return objects.toArray();
	}

}
