package mainPackage.Model;

import java.util.ArrayList;

/**
 * 
 * Klasa przechowuje bilety, które u¿ytkownik kupi b¹dŸ zarezerwuje, zale¿nie od u¿ycia
 *
 */
public class TicketCollection {
	private ArrayList<Ticket> tickets;
	
	public TicketCollection()
	{
		tickets = new ArrayList<Ticket>();
	}
	
	public void add(Ticket ticket)
	{
		tickets.add(ticket);
	}
	
	public void delete(int i)
	{
		tickets.remove(i);
	}
	
	public Ticket get(int i)
	{
		return tickets.get(i);
	}
	
	public ArrayList<Ticket> get()
	{
		return tickets;
	}
}
