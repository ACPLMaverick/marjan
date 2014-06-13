package mainPackage.Model;

import java.util.Date;
import java.util.GregorianCalendar;

/**
 * 
 * Klasa reprezentuje jeden koszt, przechowuje konkretn¹ kwotê, obiekt, którego dotyczy koszt i datê poniesienia tego kosztu.
 *
 */
public class Cost implements Comparable{
	
	private double price;
	private Object myObject;
	private Date date;
	private String type;
	
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
			price = - Math.abs(((Film) myObject).getPrice());
			this.type = "LICENCJA";
			Date currentDate = new Date();
			GregorianCalendar cal = new GregorianCalendar(currentDate.getYear(), ((currentDate.getMonth() + 1) % 12) , 1);
			this.date = cal.getTime();
		}
		else
		{
			price = 0.0;
		}
	}
	
	public double getPrice() { return price; }
	public Object getObject() { return myObject; }
	public Date getDate() { return date; }
	public String getPriceAsString() { return String.format("%.2f", this.price) + " z³"; }
	public String getType() { return type; }
	
	public int getFieldsCount() { return 3; }

	@Override
	public int compareTo(Object arg0) {
		Cost secondCost = (Cost)arg0;
		Date newDate = secondCost.getDate();
		return this.date.compareTo(newDate);
	}
	
	@Override
	public String toString()
	{
		String myString = type;
		if(type != "LICENCJA") myString += "   ";
		return myString + " | " + this.date.toString() + " | " + String.valueOf(this.price);
	}
}
