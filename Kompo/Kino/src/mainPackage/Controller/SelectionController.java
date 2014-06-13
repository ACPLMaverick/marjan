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
 * Klasa s³u¿y do konwersji kolekcji na macierze Objectów, co jest niezbêdne przy wrzucaniu danych do tabeli.
 * Odpowiada tak¿e za sortowanie i selekcjê danych po konkretnym parametrze.
 *
 */

/*
 * REPERTUAR: wybieramy po: tytu³ filmu, gatunek filmu, data filmu, 
 * KOSZTY & WYDATKI: wybieramy po: data, typ, cena
 * BILETY: wybieramy: tytu³, data, cena
 */
public interface SelectionController {
	
	
	/**
	 * 
	 * @return zwraca kolekcjê w postaci odpowiedniej macierzy Object'ów, do wrzucenia w tabelê
	 */
	public abstract Object[][] getCollectionAsObjects();
	
	/**
	 * 
	 * @return zwraca listê elementów w postaci odpowiedniej macierzy Object'ów, do wrzucenia w tabelê
	 * @param type - zwraca tylko te obiekty, które maj¹ odpowiedni typ
	 */
	public abstract Object[] getElementAsObjects(Object element);
}

//package mainPackage.Controller;
//
//import java.text.DateFormat;
//import java.text.SimpleDateFormat;
//import java.util.ArrayList;
//import java.util.Calendar;
//import java.util.Date;
//
//import mainPackage.Model.Cost;
//import mainPackage.Model.CostCollection;
//import mainPackage.Model.Model;
//import mainPackage.Model.Repertoire;
//import mainPackage.Model.Seance;
//import mainPackage.Model.Ticket;
//import mainPackage.Model.TicketCollection;
//
///**
//* 
//* Klasa s³u¿y do konwersji kolekcji na macierze Objectów, co jest niezbêdne przy wrzucaniu danych do tabeli.
//* Odpowiada tak¿e za sortowanie i selekcjê danych po konkretnym parametrze.
//*
//*/
//
///*
//* REPERTUAR: wybieramy po: tytu³ filmu, gatunek filmu, data filmu, 
//* KOSZTY & WYDATKI: wybieramy po: data, typ, cena
//* BILETY: wybieramy: tytu³, data, cena
//*/
//public class CostsSelectionController extends SelectionController {
//	
//	private Repertoire myRep;
//	private CostCollection myCC;
//	private TicketCollection myTC;
//	
//	private String paramTitle;
//	private String paramGenre;
//	private Date paramDateMin;
//	private Date paramDateMax;
//	private double paramPriceMin;
//	private double paramPriceMax;
//	
//	public CostsSelectionController(Repertoire myRep)
//	{
//		this.myRep = myRep;
//		this.myCC = null;
//		this.myTC = null;
//	}
//	
//	public CostsSelectionController(CostCollection myCC)
//	{
//		this.myCC = myCC;
//		this.myRep = null;
//		this.myTC = null;
//	}
//	
//	public CostsSelectionController(TicketCollection myTC)
//	{
//		this.myCC = null;
//		this.myRep = null;
//		this.myTC = myTC;
//	}
//	
//	/**
//	 * 
//	 * @return zwraca listê kosztów w postaci odpowiedniej macierzy Object'ów, do wrzucenia w tabelê
//	 */
//	public Object[][] getCostsAsObjects()
//	{
//		Object[][] myArray = new Object[myCC.get().size()][myCC.get().get(0).getFieldsCount()];
//		for(int i = 0; i < myCC.get().size(); i++)
//		{
//			Object[] myObjArray = getCostAsObjects(myCC.get().get(i));
//			for(int j = 0; j < myCC.get().get(i).getFieldsCount(); j++)
//			{
//				myArray[i][j] = myObjArray[j];
//			}
//		}
//		return myArray;
//	}
//	
//	/**
//	 * 
//	 * @return zwraca listê kosztów w postaci odpowiedniej macierzy Object'ów, do wrzucenia w tabelê
//	 * @param type - zwraca tylko te obiekty, które maj¹ odpowiedni typ
//	 */
//	public Object[][] getCostsAsObjects(String type)
//	{
//		ArrayList<Object[]> objAL = new ArrayList<Object[]>();
//		
//		for(Cost cost : myCC.get())
//		{
//			if(cost.getType().equals(type))
//			{
//				objAL.add(getCostAsObjects(cost));
//			}
//		}
//		
//		Object[][] myArray = new Object[objAL.size()][myCC.get().get(0).getFieldsCount()];
//		for(int i = 0; i < objAL.size(); i++)
//		{
//			Object[] myObjArray = objAL.get(i);
//			for(int j = 0; j < myCC.get().get(i).getFieldsCount(); j++)
//			{
//				myArray[i][j] = myObjArray[j];
//			}
//		}
//		
//		
//		return myArray;
//	}
//	
//	/**
//	 * 
//	 * @return zwraca listê kosztów w postaci odpowiedniej macierzy Object'ów, do wrzucenia w tabelê
//	 * @param typeMin, typeMax - zwraca tylko te obiekty, które maj¹ odpowiedni¹ cenê, w zadanym przedziale
//	 */
//	public Object[][] getCostsAsObjects(double priceMin, double priceMax)
//	{
//		ArrayList<Object[]> objAL = new ArrayList<Object[]>();
//		
//		for(Cost cost : myCC.get())
//		{
//			if(cost.getPrice() >= priceMin && cost.getPrice() <= priceMax)
//			{
//				objAL.add(getCostAsObjects(cost));
//			}
//		}
//		
//		Object[][] myArray = new Object[objAL.size()][myCC.get().get(0).getFieldsCount()];
//		for(int i = 0; i < objAL.size(); i++)
//		{
//			Object[] myObjArray = objAL.get(i);
//			for(int j = 0; j < myCC.get().get(i).getFieldsCount(); j++)
//			{
//				myArray[i][j] = myObjArray[j];
//			}
//		}
//		
//		
//		return myArray;
//	}
//	
//	/**
//	 * 
//	 * @return zwraca listê kosztów w postaci odpowiedniej macierzy Object'ów, do wrzucenia w tabelê
//	 * @param dateMin, dateMax - zwraca tylko te obiekty, które maj¹ odpowiedni¹ datê, w zadanym przedziale
//	 */
//	public Object[][] getCostsAsObjects(Date dateMin, Date dateMax)
//	{
//		ArrayList<Object[]> objAL = new ArrayList<Object[]>();
//		
//		for(Cost cost : myCC.get())
//		{
//			if(cost.getDate().getTime() >= dateMin.getTime() && cost.getDate().getTime() <= dateMax.getTime())
//			{
//				objAL.add(getCostAsObjects(cost));
//			}
//		}
//		
//		Object[][] myArray = new Object[objAL.size()][myCC.get().get(0).getFieldsCount()];
//		for(int i = 0; i < objAL.size(); i++)
//		{
//			Object[] myObjArray = objAL.get(i);
//			for(int j = 0; j < myCC.get().get(i).getFieldsCount(); j++)
//			{
//				myArray[i][j] = myObjArray[j];
//			}
//		}
//		
//		
//		return myArray;
//	}
//	
//	/**
//	 * 
//	 * @return zwraca listê biletów w postaci odpowiedniej macierzy Object'ów, do wrzucenia w tabelê
//	 */
//	public Object[][] getTicketsAsObjects()
//	{
//		Object[][] myArray = new Object[myTC.get().size()][myTC.get().get(0).getFieldsCount()];
//		for(int i = 0; i < myTC.get().size(); i++)
//		{
//			Object[] myObjArray = getTicketAsObjects(myTC.get().get(i));
//			for(int j = 0; j < myTC.get().get(i).getFieldsCount(); j++)
//			{
//				myArray[i][j] = myObjArray[j];
//			}
//		}
//		return myArray;
//	}
//	
//	/**
//	 * 
//	 * @return zwraca listê biletów w postaci odpowiedniej macierzy Object'ów, do wrzucenia w tabelê
//	 * @param title - zwraca tylko te obiekty, które maj¹ odpowiedni tytu³ seansu
//	 */
//	public Object[][] getTicketsAsObjects(String title)
//	{
//		ArrayList<Object[]> objAL = new ArrayList<Object[]>();
//		
//		for(Ticket ticket : myTC.get())
//		{
//			if(ticket.getSeance().getTitle().equals(title))
//			{
//				objAL.add(getTicketAsObjects(ticket));
//			}
//		}
//		
//		Object[][] myArray = new Object[objAL.size()][myTC.get().get(0).getFieldsCount()];
//		for(int i = 0; i < objAL.size(); i++)
//		{
//			Object[] myObjArray = objAL.get(i);
//			for(int j = 0; j < myTC.get().get(i).getFieldsCount(); j++)
//			{
//				myArray[i][j] = myObjArray[j];
//			}
//		}
//		
//		
//		return myArray;
//	}
//	
//	/**
//	 * 
//	 * @return zwraca listê biletów w postaci odpowiedniej macierzy Object'ów, do wrzucenia w tabelê
//	 * @param dateMin, dateMax - zwraca tylko te obiekty, które maj¹ odpowiedni¹ datê, w zadanym przedziale
//	 */
//	public Object[][] getTicketsAsObjects(Date dateMin, Date dateMax)
//	{
//		ArrayList<Object[]> objAL = new ArrayList<Object[]>();
//		
//		for(Ticket ticket : myTC.get())
//		{
//			if(ticket.getSeance().getDate().getTime() >= dateMin.getTime() && ticket.getSeance().getDate().getTime() <= dateMax.getTime())
//			{
//				objAL.add(getTicketAsObjects(ticket));
//			}
//		}
//		
//		Object[][] myArray = new Object[objAL.size()][myTC.get().get(0).getFieldsCount()];
//		for(int i = 0; i < objAL.size(); i++)
//		{
//			Object[] myObjArray = objAL.get(i);
//			for(int j = 0; j < myTC.get().get(i).getFieldsCount(); j++)
//			{
//				myArray[i][j] = myObjArray[j];
//			}
//		}
//		
//		
//		return myArray;
//	}
//	
//	/**
//	 * 
//	 * @return zwraca listê biletów w postaci odpowiedniej macierzy Object'ów, do wrzucenia w tabelê
//	 * @param priceMin, priceMax - zwraca tylko te obiekty, które maj¹ odpowiedni¹ cenê, w zadanym przedziale
//	 */
//	public Object[][] getTicketsAsObjects(double priceMin, double priceMax)
//	{
//		ArrayList<Object[]> objAL = new ArrayList<Object[]>();
//		
//		for(Ticket ticket : myTC.get())
//		{
//			if(ticket.getPrice() >= priceMin && ticket.getPrice() <= priceMax)
//			{
//				objAL.add(getTicketAsObjects(ticket));
//			}
//		}
//		
//		Object[][] myArray = new Object[objAL.size()][myTC.get().get(0).getFieldsCount()];
//		for(int i = 0; i < objAL.size(); i++)
//		{
//			Object[] myObjArray = objAL.get(i);
//			for(int j = 0; j < myTC.get().get(i).getFieldsCount(); j++)
//			{
//				myArray[i][j] = myObjArray[j];
//			}
//		}
//		
//		return myArray;
//	}
//	/**
//	 * 
//	 * @return zwraca repertuar w postaci odpowiedniej macierzy Object'ów, do wrzucenia w tabelê
//	 */
//	public Object[][] getRepertoireAsObjects()
//	{
//		Object[][] myArray = new Object[myRep.get().size()][myRep.get().get(0).getFieldsCount()];
//		for(int i = 0; i < myRep.get().size(); i++)
//		{
//			Object[] myObjArray = getSeanceAsObjects(myRep.get().get(i));
//			for(int j = 0; j < myRep.get().get(i).getFieldsCount(); j++)
//			{
//				myArray[i][j] = myObjArray[j];
//			}
//		}
//		return myArray;
//	}
//	
//	/**
//	 * 
//	 * @return zwraca repertuar w postaci odpowiedniej macierzy Object'ów, do wrzucenia w tabelê
//	 * @param title - zwraca tylko te obiekty, które maj¹ odpowiedni tytu³ b¹dŸ gatunek
//	 */
//	public Object[][] getRepertoireAsObjects(String title)
//	{
//		ArrayList<Object[]> objAL = new ArrayList<Object[]>();
//		
//		for(Seance seance : myRep.get())
//		{
//			if(seance.getTitle().equals(title) || seance.getGenre().equals(title))
//			{
//				objAL.add(getSeanceAsObjects(seance));
//			}
//		}
//		
//		Object[][] myArray = new Object[objAL.size()][myRep.get().get(0).getFieldsCount()];
//		for(int i = 0; i < objAL.size(); i++)
//		{
//			Object[] myObjArray = objAL.get(i);
//			for(int j = 0; j < myRep.get().get(i).getFieldsCount(); j++)
//			{
//				myArray[i][j] = myObjArray[j];
//			}
//		}
//		
//		
//		return myArray;
//	}
//	
//	/**
//	 * 
//	 * @return zwraca repertuar w postaci odpowiedniej macierzy Object'ów, do wrzucenia w tabelê
//	 * @param title - zwraca tylko te obiekty, które maj¹ odpowiedni¹ datê
//	 */
//	public Object[][] getRepertoireAsObjects(Date dateMin, Date dateMax)
//	{
//		ArrayList<Object[]> objAL = new ArrayList<Object[]>();
//		
//		for(Seance seance : myRep.get())
//		{
//			if(seance.getDate().getTime() >= dateMin.getTime() && seance.getDate().getTime() <= dateMax.getTime())
//			{
//				objAL.add(getSeanceAsObjects(seance));
//			}
//		}
//		
//		Object[][] myArray = new Object[objAL.size()][myRep.get().get(0).getFieldsCount()];
//		for(int i = 0; i < objAL.size(); i++)
//		{
//			Object[] myObjArray = objAL.get(i);
//			for(int j = 0; j < myRep.get().get(i).getFieldsCount(); j++)
//			{
//				myArray[i][j] = myObjArray[j];
//			}
//		}
//		
//		
//		return myArray;
//	}
//	
//	
//	/**
//	 * 
//	 * @return zwraca parametry seansu w tablicy Object[],
//	 * do tabeli
//	 */
//	private Object[] getSeanceAsObjects(Seance seance)
//	{
//		ArrayList<Object> objects = new ArrayList<Object>();
//		
//		DateFormat df = new SimpleDateFormat("dd-MM-yyyy HH:mm");
//		String allDate = df.format(seance.getDate());
//		String[] splitDate = allDate.split(" ");
//		
//		objects.add(seance.getTitle());
//		objects.add(splitDate[1]);
//		objects.add(splitDate[0]);
//		objects.add(String.valueOf(Model.placesAvailable - seance.getSeatPlan()));
//		objects.add(String.format("%.2f", seance.getPrice()) + " z³");
//		return objects.toArray();
//	}
//	
//	/**
//	 * 
//	 * @return zwraca parametry kosztu w tablicy Object[],
//	 * do tabeli
//	 */
//	private Object[] getCostAsObjects(Cost cost)
//	{
//		ArrayList<Object> objects = new ArrayList<Object>();
//		
//		DateFormat df = new SimpleDateFormat("dd-MM-yyyy HH:mm");
//		String allDate = df.format(cost.getDate());
//		String[] splitDate = allDate.split(" ");
//		
//		objects.add(cost.getType());
//		objects.add(splitDate[0]);
//		objects.add(String.format("%.2f", cost.getPrice()) + " z³");
//		return objects.toArray();
//	}
//	
//	/**
//	 * 
//	 * @return zwraca parametry biletu w tablicy Object[],
//	 * do tabeli
//	 */
//	private Object[] getTicketAsObjects(Ticket ticket)
//	{
//		ArrayList<Object> objects = new ArrayList<Object>();
//		
//		DateFormat df = new SimpleDateFormat("dd-MM-yyyy HH:mm");
//		String allDate = df.format(ticket.getSeance().getDate());
//		String[] splitDate = allDate.split(" ");
//		
//		objects.add(ticket.getSeance().getTitle());
//		objects.add(splitDate[0]);
//		objects.add(String.valueOf(ticket.getPrice()));
//		return objects.toArray();
//	}
//}

