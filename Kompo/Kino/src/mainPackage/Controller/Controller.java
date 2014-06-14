package mainPackage.Controller;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;

import javax.swing.JComboBox;
import javax.swing.ListSelectionModel;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import javax.swing.event.ListSelectionEvent;
import javax.swing.event.ListSelectionListener;

import mainPackage.Model.Cost;
import mainPackage.Model.Film;
import mainPackage.Model.Model;
import mainPackage.Model.Repertoire;
import mainPackage.Model.Seance;
import mainPackage.Model.Ticket;
import mainPackage.View.View;


public class Controller {
	private View theView;
	private Model theModel;
	private Date currentDate;
	
	private SelectionListener sl = new SelectionListener();			//to na razie m�j jedyny pomys� jak pobra�
																	//zaznaczony tytu�
	private TicketsSelectionListener bought = new TicketsSelectionListener();
	private BookedSelectionListener booked = new BookedSelectionListener();
	
	public Controller(View theView, Model theModel){
		this.theView = theView;
		this.theModel = theModel;
		this.currentDate = new Date();
		theModel.repertoire.connectedMode = false;					// TO JEST FLAGA, KT�RA USTAWIA, �E NIE APDEJTUJEMY BAZY DANYCH
		
		this.theView.addUserButtonListener(userButtonListener);
		this.theView.addAdminButtonListener(adminButtonListener);
		updateCosts();
	}
	
	/**
	 * zawsze generujemy ca�e koszta od nowa<br>
	 * sk�adowe koszt�w: <br>
	 * + kupione bilety, <br>
	 * - licencje film�w co miesi�c, <br>
	 * - wystawiony seans
	 */
	@SuppressWarnings("unchecked")
	public void updateCosts()
	{
		// zawsze generujemy ca�e koszta od nowa
		// sk�adowe koszt�w: + kupione bilety, - licencje film�w co miesi�c, - wystawiony seans
		theModel.costs.get().clear();
		
		for(Seance seance : theModel.repertoire.get())
		{
			if(seance.getDate().getTime() < currentDate.getTime())theModel.costs.add(new Cost(seance));
		}
		
		for(Ticket ticket : theModel.boughtTickets.get())
		{
			theModel.costs.add(new Cost(ticket));
		}
		
		for(Film film : theModel.repertoire.getFilms())
		{
			theModel.costs.add(new Cost(film));
		}
		
		Collections.sort(theModel.costs.get());
	}
	
	/**
	 * kasujemy wszystkie interesujace nas elementy (kupione/zarezerwowane bilety, koszta, seanse)<br>
	 * starsze ni� podana data
	 * @param mode - 0: wszystko, 1: bez repertuaru
	 */
	public void clearAllByDate(Date dateMin, int mode)
	{
		for(int i = 0; i < theModel.boughtTickets.get().size(); i++)
		{
			Ticket currentTicket = theModel.boughtTickets.get(i);
			if(currentTicket.getSeance().getDate().getTime() < dateMin.getTime()) theModel.boughtTickets.delete(i);
		}
		
		for(int i = 0; i < theModel.reservedTickets.get().size(); i++)
		{
			Ticket currentTicket = theModel.reservedTickets.get(i);
			if(currentTicket.getSeance().getDate().getTime() < dateMin.getTime()) theModel.reservedTickets.delete(i);
		}
		
		for(int i = 0; i < theModel.costs.get().size(); i++)
		{
			Cost currentCost = theModel.costs.get(i);
			if(currentCost.getDate().getTime() < dateMin.getTime()) theModel.costs.delete(i);
		}
		
		if(mode != 1)
		{
			for(int i = 0; i < theModel.repertoire.get().size(); i++)
			{
				Seance seance = theModel.repertoire.get(i);
				if(seance.getDate().getTime() < dateMin.getTime()) theModel.repertoire.delete(i);
			}
		}
	}
	
	public void saveCostsToFile()
	{
		String path = theView.createSaveMenu("txt");
		if(path == null) return;
		
		File myFile = new File(path);
		
		try
		{
			myFile.createNewFile();
			FileWriter stream = new FileWriter(myFile);
			stream.write(theModel.costs.toString());
			stream.close();
		}
		catch(IOException e)
		{
			System.out.println(e.getMessage());
		}
	}
	
	public void updateRepertoireTable()
	{
		SelectionController updater = new RepertoireSelectionController(theModel.repertoire, null, null, null, null, 0, 0);
		Object[][] newContent = updater.getCollectionAsObjects();
		theView.um.setTableContent(newContent);
	}
	
	public void updateBoughtTable()
	{
		SelectionController updater = new TicketsSelectionController(theModel.boughtTickets);
		Object[][] newContent = updater.getCollectionAsObjects();
		theView.um.basketMenu.setBoughtTableContent(newContent);
	}

	public void updateBookedTable()
	{
		SelectionController updater = new TicketsSelectionController(theModel.reservedTickets);
		Object[][] newContent = updater.getCollectionAsObjects();
		theView.um.basketMenu.setBookedTableContent(newContent);
	}
	
	public void updateCostsTable()
	{
		SelectionController updater = new CostsSelectionController(theModel.costs, null, null, null, 0, 0);
		Object[][] newContent = updater.getCollectionAsObjects();
		//theView.um.setTableContent(newContent);
	}
	
	public void serialiseRepertoire()
	{
		String path = theView.createSaveMenu("xml");
		if(path == null) return;
		SerializationController<Repertoire> ser = new SerializationController<Repertoire>(theModel.repertoire);
		ser.serialize(path);
	}
	
	public void deserialiseRepertoire()
	{
		String path = theView.createSaveMenu("xml");
		if(path == null) return;
		SerializationController<Repertoire> ser = new SerializationController<Repertoire>(theModel.repertoire);
		theModel.repertoire = (Repertoire)ser.deserialize(path);
	}
	
	public void createChart()
	{
		SelectionController contr = new CostsSelectionController(theModel.costs);
		ArrayList<ArrayList<Number>> data = contr.getCollectionAsChartData();
		theView.createCostChart(data.get(0), data.get(1));
	}
	
	private void createReminder()
	{
		int costCount = checkIfReminderIsNeeded();
		if(costCount == 0) return;
		String remTitle = String.format("<html><div style=\"width:%dpx;\">%s</div><html>",300, "Przypomnienie o <br/>p�atno�ci za licencj�!");
		String remDesc = String.format("<html><div style=\"width:%dpx;\">%s</div><html>",250, 
						"Termin op�aty " + String.valueOf(costCount) + " licencji na filmy jest mniejszy ni� 24 godziny b�d� ju� up�yn��!\n "
						+ "Zaleca si� jak najszybsze dokonanie op�at!");
		theView.createSmallWindow(remTitle, remDesc);
	}

	private int checkIfReminderIsNeeded()
	{
		ArrayList<Cost> tempCosts = new ArrayList<Cost>();
		for(Cost cost : theModel.costs.get())
		{
			if((cost.getDate().getTime() >= this.currentDate.getTime() - 86400000) && cost.getType().equals("LICENSE")) tempCosts.add(cost);
		}
		if(tempCosts.size() > 0) return tempCosts.size();
		else return 0;
	}
	
	ActionListener userButtonListener = new ActionListener(){
		public void actionPerformed(ActionEvent e){
			theView.setVisible(false);
			theView.createUserMenu();
			theView.um.addBuyButtonListener(buyButtonListener);
			theView.um.disableButton(theView.um.getBuyButton());
			theView.um.addBookButtonListener(bookButtonListener);
			theView.um.disableButton(theView.um.getBookButton());
			theView.um.addBackButtonListener(backButtonListener);
			theView.um.addBasketButtonListener(basketButtonListener);
			theView.um.disableButton(theView.um.getBasketButton());
			theView.um.addFilterComboListener(filterComboListener);
			if(theModel.boughtTickets.get().isEmpty() == false)
			{
				theView.um.enableButton(theView.um.getBasketButton());
			}
			theView.um.getUserListSelection().addListSelectionListener(sl);
			updateRepertoireTable();
		}
	};
	
	ActionListener adminButtonListener = new ActionListener(){
		public void actionPerformed(ActionEvent e){
			theView.setVisible(false);
			theView.createAdminMenu();
			createReminder();
			createChart(); //TEMP
		}
	};
	
	ActionListener buyButtonListener = new ActionListener(){
		public void actionPerformed(ActionEvent e){
			theView.um.createBuyTicketMenu();
			theView.um.buyTicket.getTicketCount().addChangeListener(spinnerChangeListener);
			theView.um.buyTicket.setSeanceTitle(sl.seance.getTitle(), sl.seance.getDateAsString());
			theView.um.buyTicket.setTicketPrice(sl.seance.getPrice());
			theView.um.buyTicket.addBuyButtonListener(buyTicketButtonListener);
		}
	};
	
	ChangeListener spinnerChangeListener = new ChangeListener(){
		@Override
		public void stateChanged(ChangeEvent arg0) {
			double cena = sl.seance.getPrice();
			cena *= Double.valueOf(theView.um.buyTicket.getSpinnerListModel().getValue().toString());
			theView.um.buyTicket.setTicketPrice(cena);
		}
	};
	
	ActionListener buyTicketButtonListener = new ActionListener(){
		public void actionPerformed(ActionEvent e){
			theView.createSmallWindow("Zakup trafi� do koszyka");
			for(int i = 0; i<Integer.valueOf(theView.um.buyTicket.getSpinnerListModel().getValue().toString()); i++){
				theModel.boughtTickets.add(new Ticket(sl.seance));
			}
			theView.um.enableButton(theView.um.getBasketButton());
		}
	};
	
	ActionListener bookButtonListener = new ActionListener(){
		public void actionPerformed(ActionEvent e){
			theView.um.createBookTicketMenu();
			theView.um.bookTicket.getTicketCount().addChangeListener(spinnerBookChangeListener);
			theView.um.bookTicket.setSeanceTitle(sl.seance.getTitle(), sl.seance.getDateAsString());
			theView.um.bookTicket.setTicketPrice(sl.seance.getPrice());
			theView.um.bookTicket.addBookButtonListener(bookTicketButtonListener);
		}
	};
	
	ChangeListener spinnerBookChangeListener = new ChangeListener(){
		@Override
		public void stateChanged(ChangeEvent arg0) {
			double cena = sl.seance.getPrice();
			cena *= Double.valueOf(theView.um.bookTicket.getSpinnerListModel().getValue().toString());
			theView.um.bookTicket.setTicketPrice(cena);
		}
	};
	
	ActionListener bookTicketButtonListener = new ActionListener(){
		public void actionPerformed(ActionEvent e){
			theView.createSmallWindow("Rezerwacja trafi�a do koszyka");
			for(int i = 0; i<Integer.valueOf(theView.um.bookTicket.getSpinnerListModel().getValue().toString()); i++){
				theModel.reservedTickets.add(new Ticket(sl.seance));
			}
			theView.um.enableButton(theView.um.getBasketButton());
		}
	};
	
	ActionListener backButtonListener = new ActionListener(){
		public void actionPerformed(ActionEvent e){
			theView.setVisible(true);
			theView.um.getUserMenu().setVisible(false);
		}
	};
	
	ActionListener basketButtonListener = new ActionListener(){
		public void actionPerformed(ActionEvent e){
			theView.um.createBasketMenu();
			theView.um.basketMenu.getBoughtTicketsListSelection().addListSelectionListener(bought);
			theView.um.basketMenu.getBookedTicketsListSelection().addListSelectionListener(booked);
			theView.um.basketMenu.addDeleteTicketButtonListener(deleteTicketButtonListener);
			theView.um.basketMenu.addDeleteReservationButtonListener(deleteReservationButtonListener);
			theView.um.disableButton(theView.um.basketMenu.getBoughtDeleteButton());
			theView.um.disableButton(theView.um.basketMenu.getBookedDeleteButton());
			if(theModel.boughtTickets.get().size() == 0)
				updateBookedTable();
			else if(theModel.reservedTickets.get().size() == 0)
				updateBoughtTable();
			else if(theModel.boughtTickets.get().size() != 0 && theModel.reservedTickets.get().size() != 0){
				updateBookedTable();
				updateBoughtTable();
			}
		}
	};
	
	ActionListener filterComboListener = new ActionListener(){
		public void actionPerformed(ActionEvent e){
			JComboBox cb = (JComboBox)e.getSource();
			String filter = (String)cb.getSelectedItem();
			if(filter == "Gatunek"){
				theView.um.getGenreFilterCombo().setEnabled(true);
				
			}
			else theView.um.getGenreFilterCombo().setEnabled(false);
		}
	};
	
	ActionListener deleteTicketButtonListener = new ActionListener(){
		public void actionPerformed(ActionEvent e){
			theModel.boughtTickets.delete(bought.row);
			if(theModel.boughtTickets.get().size() == 0){
				theView.um.basketMenu.getBasketFrame().dispose();
				theModel.boughtTickets.add(new Ticket(new Seance()));
				//theView.um.disableButton(theView.um.getBasketButton());
			}
			else{
				theView.um.basketMenu.getBasketFrame().setVisible(false);
				updateBoughtTable();
				theView.um.basketMenu.getBasketFrame().setVisible(true);
			}
		}
	};
	
	ActionListener deleteReservationButtonListener = new ActionListener(){
		public void actionPerformed(ActionEvent e){
			theModel.reservedTickets.delete(booked.row);
			if(theModel.reservedTickets.get().size() == 0){
				theView.um.basketMenu.getBasketFrame().dispose();
				theModel.reservedTickets.add(new Ticket(new Seance()));
				//theView.um.disableButton(theView.um.getBasketButton());
			}
			else{
				theView.um.basketMenu.getBasketFrame().setVisible(false);
				updateBookedTable();
				theView.um.basketMenu.getBasketFrame().setVisible(true);
			}
		}
	};
	
	class SelectionListener implements ListSelectionListener {
		public Seance seance;
		@Override
		public void valueChanged(ListSelectionEvent e) {
			ListSelectionModel lsm = (ListSelectionModel)e.getSource();
			// TODO Auto-generated method stub
			if(e.getValueIsAdjusting()) return;
			int row = theView.um.getUserTable().getSelectedRow();
			if(row < 0) return;
			
			theView.um.enableButton(theView.um.getBuyButton());
			theView.um.enableButton(theView.um.getBookButton());
			seance = theModel.repertoire.get(row);
		}
	};
	
	class TicketsSelectionListener implements ListSelectionListener{
		public Ticket ticket;
		public int row;
		@Override
		public void valueChanged(ListSelectionEvent e) {
			// TODO Auto-generated method stub
			if(e.getValueIsAdjusting()) return;
			row = theView.um.basketMenu.getTicketsTable().getSelectedRow();
			if(row < 0) return;
			
			if(theModel.boughtTickets.get().size() != 0)
				theView.um.enableButton(theView.um.basketMenu.getBoughtDeleteButton());
			System.out.println(Integer.valueOf(row));
		}
	};
	
	class BookedSelectionListener implements ListSelectionListener{
		public int row;
		@Override
		public void valueChanged(ListSelectionEvent e) {
			// TODO Auto-generated method stub
			if(e.getValueIsAdjusting()) return;
			row = theView.um.basketMenu.getBookedTable().getSelectedRow();
			if(row < 0) return;
			
			if(theModel.reservedTickets.get().size() != 0)
				theView.um.enableButton(theView.um.basketMenu.getBookedDeleteButton());
			System.out.println(Integer.valueOf(row));
		}
	};
}
	
