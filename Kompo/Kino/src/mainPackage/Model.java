package mainPackage;

/**
 * 
 * 
 * Definiuje model danych aplikacji, przechowuje wszystkie potrzebne dane i obs�uguje ��czno�� z baz� danych
 */
public class Model {
	// repertuar, kupione bilety, zarezerwowane bilety, koszta, wydatki
	private Controller myController;
	
	// TODO: sprawdzi� czy faktycznie musza by� public
	public Repertoire repertoire;
	public TicketCollection boughtTickets;
	public TicketCollection reservedTickets;
	public CostCollection costs;
	public DBController dataBaseController;
	
	public Model(Controller controller)
	{
		this.myController = controller;
		
		this.repertoire = new Repertoire();
		this.boughtTickets = new TicketCollection();
		this.reservedTickets = new TicketCollection();
		this.costs = new CostCollection();
		this.dataBaseController = new DBController();
		
		dataBaseController.getWholeRepertoire();
	}
}
