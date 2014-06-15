package mainPackage.Model;

/**
 * 
 * Klasa odpowiada za reprezentację jednego biletu
 *
 */
public class Ticket {
	private Seance mySeance;
	private double price;
	
	public static final int fieldsCount = 3;
	
	public Ticket(Seance mySeance)
	{
		this.mySeance = mySeance;
		this.price = this.mySeance.getPrice();
	}
	
	public Seance getSeance()
	{
		return this.mySeance;
	}
	
	public double getPrice()
	{
		return this.price;
	}
}
