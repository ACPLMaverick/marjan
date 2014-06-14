package mainPackage.Controller;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;

import mainPackage.Model.Ticket;
import mainPackage.Model.TicketCollection;

public class TicketsSelectionController implements SelectionController {

	private TicketCollection myCollection;
	
	private String paramTitle;
	private Date paramDateMin;
	private Date paramDateMax;
	private double paramPriceMin;
	private double paramPriceMax;
	
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
	 * 
	 * @param myRep obiekt Repertoire
	 * @param paramTitle ¿¹dany typ kosztu albo null
	 * @param paramDateMin ¿¹dana data minimalna albo null
	 * @param paramDateMax ¿¹dana data maksymalna albo null
	 * @param paramPriceMin ¿¹dana cena minimalna albo 0
	 * @param paramPriceMax ¿¹dana cena maksymalna albo 0
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
	 * 
	 * @return zwraca listê kosztów w postaci odpowiedniej macierzy Object'ów, do wrzucenia w tabelê
	 * @param Title - zwraca tylko te obiekty, które maj¹ odpowiedni typ
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
	
	@Override
	public ArrayList<ArrayList<Number>> getCollectionAsChartData() {
		// TODO Auto-generated method stub
		return null;
	}
	
	
	/**
	 * 
	 * @return zwraca parametry kosztu w tablicy Object[],
	 * do tabeli
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
