package mainPackage.Model;

// TODO: Auto-generated Javadoc
/**
 * Klasa odpowiada za reprezentacjê jednego biletu.
 */
public class Ticket {

	private Seance mySeance;
	private double price;
	
	/** Stala okreslajaca ilosc pol jakie opisuja bilet w interfejsie. */
	public static final int fieldsCount = 3;
	
	/**
	 * Tworzy nowy obiekt typu Ticket.
	 *
	 * @param mySeance seans, na ktory jest bilet.
	 */
	public Ticket(Seance mySeance)
	{
		this.mySeance = mySeance;
		this.price = this.mySeance.getPrice();
	}
	
	/**
	 * Zwraca seans.
	 *
	 * @return Seans.
	 */
	public Seance getSeance()
	{
		return this.mySeance;
	}
	
	/**
	 * Zwraca cene biletu.
	 *
	 * @return Cene biletu.
	 */
	public double getPrice()
	{
		return this.price;
	}
}
