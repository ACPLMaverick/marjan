package mainPackage.Model;

import java.util.Date;

import mainPackage.Controller.Controller;

// TODO: Auto-generated Javadoc
/**
 * Klasa definiuje model danych aplikacji, przechowuje wszystkie potrzebne dane i obs³uguje ³¹cznoœæ z baz¹ danych.
 */
public class Model {
	// repertuar, kupione bilety, zarezerwowane bilety, koszta, wydatki
	private Controller myController;
	
	/** Pole typu Repertoire przechowujace obecny repertuar kina. */
	public Repertoire repertoire;
	
	/** Kolekcja kupionych biletow. */
	public TicketCollection boughtTickets;
	
	/** Kolekcja zarezerwowanych biletow. */
	public TicketCollection reservedTickets;
	
	/** Kolekcja zyskow i wydatkow. */
	public CostCollection costs;
	
	/** Stala okreslajaca ilosc wolnych miejsc na kazdy seans. */
	public static final int placesAvailable = 60;
	
	/**
	 * Tworzy nowy obiekt typu Model spelniajacy zalozenia warstwy danych architektury MVC.
	 *
	 * @param controller referencja do obiektu typu Controller, przetwarzajacej dane.
	 */
	public Model(Controller controller)
	{
		this.myController = controller;
		
		this.boughtTickets = new TicketCollection();
		this.reservedTickets = new TicketCollection();
		this.costs = new CostCollection();
		this.repertoire = new Repertoire();
	}
}
