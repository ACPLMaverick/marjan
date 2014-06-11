package mainPackage;

import java.util.Date;

/**
 * 
 * Klasa reprezentuje jeden koszt, przechowuje konkretn� kwot�, obiekt, kt�rego dotyczy koszt i dat� poniesienia tego kosztu.
 *
 */
public class Cost {
	
	private double price;
	private Object myObject;
	private Date date;
	
	public Cost(Object myObject)
	{
		this.myObject = myObject;
		this.date = new Date();
		
		//TODO: bilet, seans, licencja, 
		if(myObject instanceof Ticket)
		{
			price = Math.abs(((Ticket) myObject).getPrice());
		}
		else if(myObject instanceof Seance)
		{
			price = -100.0;
			// fixed cost of having a seance
		}
		else
		{
			price = 0.0;
		}
	}
	
	public double getPrice() { return price; }
	public Object getObject() { return myObject; }
	public Date getDate() { return date; }
}
