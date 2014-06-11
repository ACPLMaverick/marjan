package mainPackage;

/**
 * 
 * 
 * Definiuje model danych aplikacji, przechowuje wszystkie potrzebne dane i obs³uguje ³¹cznoœæ z baz¹ danych
 */
public class Model {
	// repertuar, kupione bilety, zarezerwowane bilety, koszta, wydatki
	private Controller myController;
	
	public Repertoire repertoire;
	public TicketCollection boughtTickets;
	public TicketCollection reservedTickets;
	public CostCollection costs;
	
	public Model(Controller controller)
	{
		this.myController = controller;
		
		this.repertoire = new Repertoire();
		this.boughtTickets = new TicketCollection();
		this.reservedTickets = new TicketCollection();
		this.costs = new CostCollection();
	}
}
