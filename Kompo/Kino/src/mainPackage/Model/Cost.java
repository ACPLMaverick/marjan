package mainPackage.Model;

import java.util.Date;
import java.util.GregorianCalendar;
import java.util.TimeZone;

// TODO: Auto-generated Javadoc
/**
 * 
 * Klasa reprezentuje jeden koszt, przechowuje konkretna kwote, obiekt, ktorego dotyczy koszt date poniesienia tego kosztu.
 * oraz rodzaj kosztu: BILET, SEANS lub LICENCJA
 * 
 */
public class Cost implements Comparable{
	
	private double price;
	private Object myObject;
	private Date date;
	private String type;
	
	/** Stala okreslajaca ilosc pol jakie opisuja koszt w interfejsie. */
	public static final int fieldsCount = 3;
	
	/**
	 * Tworzy nowy obiekt typu Cost.
	 *
	 * @param myObject obiekt, ktory decyduje o typie kosztu. Moze to byc bilet, seans albo film.
	 */
	@SuppressWarnings("deprecation")
	public Cost(Object myObject)
	{
		this.myObject = myObject;
		
		//TODO: bilet, seans, licencja, 
		if(myObject instanceof Ticket)
		{
			price = Math.abs(((Ticket) myObject).getPrice());
			this.type = "BILET";
			this.date = ((Ticket) myObject).getSeance().getDate();
		}
		else if(myObject instanceof Seance)
		{
			price = -100.0;
			// fixed cost of having a seance
			this.type = "SEANS";
			this.date = ((Seance) myObject).getDate();
		}
		else if(myObject instanceof Film)
		{
			price = - Math.abs(((Film) myObject).getLicensePrice());
			this.type = "LICENCJA";
			Date currentDate = new Date();
			GregorianCalendar cal = new GregorianCalendar(currentDate.getYear() + 1900, ((currentDate.getMonth() + 1) % 12) , 1);
			cal.setTimeZone(TimeZone.getTimeZone("PST"));
			this.date = cal.getTime();
		}
		else
		{
			price = 0.0;
		}
	}
	
	/**
	 * Zwraca cene.
	 *
	 * @return cene.
	 */
	public double getPrice() { return price; }
	
	/**
	 * Zwraca bilet, seans lub film, w zaleznosci od typu kosztu.
	 *
	 * @return obiekt.
	 */
	public Object getObject() { return myObject; }
	
	/**
	 * Zwraca date zaksiegowania kosztu.
	 *
	 * @return date.
	 */
	public Date getDate() { return date; }
	
	/**
	 * Zwraca cene.
	 *
	 * @return cene typu String.
	 */
	public String getPriceAsString() { return String.format("%.2f", this.price) + " z³"; }
	
	/**
	 * Zwraca typ kosztu.
	 *
	 * @return typ: BILET, SEANS lub LICENCJA.
	 */
	public String getType() { return type; }

	/* (non-Javadoc)
	 * @see java.lang.Comparable#compareTo(java.lang.Object)
	 */
	@Override
	public int compareTo(Object arg0) {
		Cost secondCost = (Cost)arg0;
		Date newDate = secondCost.getDate();
		return this.date.compareTo(newDate);
	}
	
	/* (non-Javadoc)
	 * @see java.lang.Object#toString()
	 */
	@Override
	public String toString()
	{
		String myString = type;
		if(type != "LICENCJA") myString += "   ";
		return myString + " | " + this.date.toString() + " | " + String.valueOf(this.price);
	}
}
