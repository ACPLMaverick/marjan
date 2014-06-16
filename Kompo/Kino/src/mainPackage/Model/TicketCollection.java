package mainPackage.Model;

import java.util.ArrayList;

// TODO: Auto-generated Javadoc
/**
 * Klasa przechowuje bilety, które u¿ytkownik kupi b¹dŸ zarezerwuje, zale¿nie od u¿ycia.
 */
public class TicketCollection {

	private ArrayList<Ticket> tickets;
	
	/**
	 * Tworzy nowy obiekt typu TicketCollection.
	 */
	public TicketCollection()
	{
		tickets = new ArrayList<Ticket>();
	}
	
	/**
	 * Dodaje nowy bilet do listy biletow.
	 *
	 * @param ticket bilet.
	 */
	public void add(Ticket ticket)
	{
		tickets.add(ticket);
	}
	
	/**
	 * Usuwa bilet o podanym indeksie z listy biletow.
	 *
	 * @param i indeks w liscie biletow.
	 */
	public void delete(int i)
	{
		tickets.remove(i);
	}
	
	/**
	 * Zwraca bilet o podanym indeksie z listy biletow.
	 *
	 * @param i indeks w liscie biletow.
	 * @return Bilet o podanym indeksie.
	 */
	public Ticket get(int i)
	{
		return tickets.get(i);
	}
	
	/**
	 * Zwraca cala liste biletow.
	 *
	 * @return Cala liste biletow.
	 */
	public ArrayList<Ticket> get()
	{
		return tickets;
	}
}
