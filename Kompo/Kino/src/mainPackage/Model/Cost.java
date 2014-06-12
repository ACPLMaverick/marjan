package mainPackage.Model;

import java.util.Date;

/**
 * 
 * Klasa reprezentuje jeden koszt, przechowuje konkretn¹ kwotê, obiekt, którego dotyczy koszt i datê poniesienia tego kosztu.
 *
 */
public class Cost {
	
	private double price;
	private Object myObject;
	private Date date;
	private String type;
	
	public Cost(Object myObject)
	{
		this.myObject = myObject;
		this.date = new Date();
		
		//TODO: bilet, seans, licencja, 
		if(myObject instanceof Ticket)
		{
			price = Math.abs(((Ticket) myObject).getPrice());
			this.type = "BILET";
		}
		else if(myObject instanceof Seance)
		{
			price = -100.0;
			// fixed cost of having a seance
			this.type = "SEANS";
		}
		else if(myObject instanceof License)
		{
			price = - Math.abs(((License) myObject).getPrice());
			this.type = "LICENCJA";
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
}
