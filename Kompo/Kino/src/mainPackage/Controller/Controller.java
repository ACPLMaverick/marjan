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
import java.util.GregorianCalendar;

import javax.swing.JComboBox;
import javax.swing.ListSelectionModel;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import javax.swing.event.ListSelectionEvent;
import javax.swing.event.ListSelectionListener;
import javax.xml.transform.OutputKeys;

import mainPackage.Model.Cost;
import mainPackage.Model.Film;
import mainPackage.Model.Model;
import mainPackage.Model.Repertoire;
import mainPackage.Model.Seance;
import mainPackage.Model.Ticket;
import mainPackage.View.UserMenu;
import mainPackage.View.UserMenuAdmin;
import mainPackage.View.View;


public class Controller {
	private View theView;
	private Model theModel;
	private Date currentDate;
	
	private SelectionListener sl = new SelectionListener();			//to na razie mój jedyny pomys³ jak pobraæ
																	//zaznaczony tytu³
	private CostsSelectionListener slCosts = new CostsSelectionListener();
	private FilmsSelectionListener slFilms = new FilmsSelectionListener();
	
	private TicketsSelectionListener bought = new TicketsSelectionListener();
	private BookedSelectionListener booked = new BookedSelectionListener();
	private UserMenu currentMenu;
	private int seatPlan;	//póki co tutaj - ogólna iloœæ miejsc zarówno dla buy i book
	
	public Controller() { }
	
	public Controller(View theView, Model theModel){
		this.theView = theView;
		this.theModel = theModel;
		this.currentDate = new Date();
		theModel.repertoire.connectedMode = false;					// TO JEST FLAGA, KTÓRA USTAWIA, ¯E NIE APDEJTUJEMY BAZY DANYCH
		
		this.theView.addUserButtonListener(userButtonListener);
		this.theView.addAdminButtonListener(adminButtonListener);
		
		updateCosts();
	}
	
	/**
	 * zawsze generujemy ca³e koszta od nowa<br>
	 * sk³adowe kosztów: <br>
	 * + kupione bilety, <br>
	 * - licencje filmów co miesi¹c, <br>
	 * - wystawiony seans
	 */
	@SuppressWarnings("unchecked")
	private void updateCosts()
	{
		// zawsze generujemy ca³e koszta od nowa
		// sk³adowe kosztów: + kupione bilety, - licencje filmów co miesi¹c, - wystawiony seans
		theModel.costs.get().clear();
		
		for(Seance seance : theModel.repertoire.get())
		{
//			if(seance.getDate().getTime() < currentDate.getTime())theModel.costs.add(new Cost(seance));
			theModel.costs.add(new Cost(seance));
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
	 * starsze ni¿ podana data
	 * @param mode - 0: wszystko, 1: bez repertuaru
	 */
	private void clearAllByDate()
	{
		UserMenuAdmin admin = (UserMenuAdmin) currentMenu;
		ArrayList<String> contentCost = admin.getAllFilterContentOfCost();
		String costType = contentCost.get(0);
		Date costdateMin;
		Date costdateMax;
		double costpriceMin;
		double costpriceMax;
		
		GregorianCalendar costcalMin = new GregorianCalendar(Integer.valueOf(contentCost.get(3)), Integer.valueOf(contentCost.get(2)) - 1, 
				Integer.valueOf(contentCost.get(1)), 0, 0);
		GregorianCalendar costcalMax = new GregorianCalendar(Integer.valueOf(contentCost.get(6)), Integer.valueOf(contentCost.get(5)) - 1, 
				Integer.valueOf(contentCost.get(4)), 23, 59);
		costdateMin = costcalMin.getTime();
		costdateMax = costcalMax.getTime();
		
		costpriceMin = Double.valueOf(contentCost.get(7));
		costpriceMax = Double.valueOf(contentCost.get(8));
		if(costpriceMax == 0) costpriceMax = Double.MAX_VALUE;
		if(costpriceMin == 0) costpriceMin = -Double.MAX_VALUE;
		
		ArrayList<Seance> tbdSeance = new ArrayList<Seance>();
		ArrayList<Ticket> tbdBoughtTicket = new ArrayList<Ticket>();
		ArrayList<Ticket> tbdReservedTicket = new ArrayList<Ticket>();
		ArrayList<Film> tbdFilm = new ArrayList<Film>();
		
		for(int i = 0; i < theModel.boughtTickets.get().size(); i++)
		{
			Ticket currentTicket = theModel.boughtTickets.get(i);
			if((costType.equals("SEANS") || costType.equals("LICENCJA")) || 
					currentTicket.getSeance().getDate().getTime() < costdateMin.getTime() ||
					currentTicket.getSeance().getDate().getTime() > costdateMax.getTime() ||
					currentTicket.getSeance().getPrice() < costpriceMin ||
					currentTicket.getSeance().getPrice() > costpriceMax
					) tbdBoughtTicket.add(theModel.boughtTickets.get(i));
		}
		
		for(int i = 0; i < theModel.reservedTickets.get().size(); i++)
		{
			Ticket currentTicket = theModel.reservedTickets.get(i);
			if((costType.equals("SEANS") || costType.equals("LICENCJA")) || 
					currentTicket.getSeance().getDate().getTime() < costdateMin.getTime() ||
					currentTicket.getSeance().getDate().getTime() > costdateMax.getTime() ||
					currentTicket.getSeance().getPrice() < costpriceMin ||
					currentTicket.getSeance().getPrice() > costpriceMax
					) tbdReservedTicket.add(theModel.reservedTickets.get(i));
		}
		
		for(int i = 0; i < theModel.repertoire.get().size(); i++)
		{
			Seance currentSeance = theModel.repertoire.get(i);
			if((costType.equals("LICENCJA") || costType.equals("BILET")) || 
					currentSeance.getDate().getTime() < costdateMin.getTime() ||
					currentSeance.getDate().getTime() > costdateMax.getTime() ||
					currentSeance.getPrice() < costpriceMin ||
					currentSeance.getPrice() > costpriceMax
					) tbdSeance.add(theModel.repertoire.get(i));
		}
		
		for(int i = 0; i < theModel.repertoire.getFilms().size(); i++)
		{
			Film currentFilm = theModel.repertoire.getFilm(i);
			if((costType.equals("SEANS") || costType.equals("BILET")) || 
					currentFilm.getPrice() < costpriceMin ||
					currentFilm.getPrice() > costpriceMax
					) tbdFilm.add(theModel.repertoire.getFilms().get(i));
		}
		
		theModel.boughtTickets.get().removeAll(tbdBoughtTicket);
		theModel.reservedTickets.get().removeAll(tbdReservedTicket);
		theModel.repertoire.get().removeAll(tbdSeance);
		theModel.repertoire.getFilms().removeAll(tbdFilm);
		
		updateCosts();
		updateFiltersInView();
	}
	
	private void saveCostsToFile()
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
	
	private void updateRepertoireTable(String title, String genre, Date dateMin, Date dateMax, double priceMin, double priceMax)
	{
		SelectionController updater = new RepertoireSelectionController(theModel.repertoire, title, genre, dateMin, dateMax, priceMin, priceMax);
		Object[][] newContent = updater.getCollectionAsObjects();
		currentMenu.setTableContent(newContent);
	}
	
	private void updateBoughtTable()
	{
		SelectionController updater = new TicketsSelectionController(theModel.boughtTickets);
		Object[][] newContent = updater.getCollectionAsObjects();
		currentMenu.basketMenu.setBoughtTableContent(newContent);
	}

	private void updateBookedTable()
	{
		SelectionController updater = new TicketsSelectionController(theModel.reservedTickets);
		Object[][] newContent = updater.getCollectionAsObjects();
		currentMenu.basketMenu.setBookedTableContent(newContent);
	}
	
	private void updateFilmsTable()
	{
		SelectionController updater = new FilmsSelectionController(theModel.repertoire.getFilms());
		Object[][] newContent = updater.getCollectionAsObjects();
		if(currentMenu instanceof UserMenuAdmin)
		{
			((UserMenuAdmin) currentMenu).setTableContentOfFilms(newContent);
		}
	}
	
	private void updateCostsTable(String type, Date dateMin, Date dateMax, double priceMin, double priceMax)
	{
		SelectionController updater = new CostsSelectionController(theModel.costs, type, dateMin, dateMax, priceMin, priceMax);
		Object[][] newContent = updater.getCollectionAsObjects();
		if(currentMenu instanceof UserMenuAdmin)
		{
			((UserMenuAdmin) currentMenu).setTableContentOfCost(newContent);
		}
		//theView.um.setTableContent(newContent);
	}
	
	private void serialiseRepertoire()
	{
		String path = theView.createSaveMenu("xml");
		if(path == null) return;
		SerializationController<Repertoire> ser = new SerializationController<Repertoire>(theModel.repertoire);
		ser.serialize(path);
	}
	
	private void deserialiseRepertoire()
	{
		String path = theView.createLoadMenu();
		if(path == null) return;
		SerializationController<Repertoire> ser = new SerializationController<Repertoire>(theModel.repertoire);
		Repertoire newRep = (Repertoire)ser.deserialize(path);
		theModel.repertoire.get().clear();
		theModel.repertoire.get().addAll(newRep.get());
	}
	
	private void createChart()
	{
		SelectionController contr = new CostsSelectionController(theModel.costs);
		ArrayList<ArrayList<Number>> data = contr.getCollectionAsChartData();
		theView.createCostChart(data.get(0), data.get(1));
	}
	
	public Date getCurrentDate() { return this.currentDate; }
	
	public ArrayList<String> getFilmTitles()
	{
		ArrayList<String> filmTitles = new ArrayList<String>();
		for(Film film : theModel.repertoire.getFilms())
		{
			filmTitles.add(film.getTitle());
		}
		return filmTitles;
	}
	
	private void createReminder()
	{
		int costCount = checkIfReminderIsNeeded();
		if(costCount == 0) return;
		String remTitle = String.format("<html><div style=\"width:%dpx;\">%s</div><html>",300, "Przypomnienie o <br/>p³atnoœci za licencjê!");
		String remDesc = String.format("<html><div style=\"width:%dpx;\">%s</div><html>",250, 
						"Termin op³aty " + String.valueOf(costCount) + " licencji na filmy jest mniejszy ni¿ 24 godziny b¹dŸ ju¿ up³yn¹³!\n "
						+ "Zaleca siê jak najszybsze dokonanie op³at!");
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
	
	private void updateFiltersInView()
	{
		ArrayList<String> content = currentMenu.getAllFilterContent();
		String title = "";
		String genre = "";
		Date dateMin;
		Date dateMax;
		double priceMin;
		double priceMax;
		
		if(content.get(0) == "wszystkie filmy") title = "";
		else title = content.get(0);
		
		if(content.get(1) == "wszystkie gatunki") genre = "";
		else genre = content.get(1);
		
		GregorianCalendar calMin = new GregorianCalendar(Integer.valueOf(content.get(4)), Integer.valueOf(content.get(3)) - 1, 
				Integer.valueOf(content.get(2)), 0, 0);
		GregorianCalendar calMax = new GregorianCalendar(Integer.valueOf(content.get(7)), Integer.valueOf(content.get(6)) - 1, 
				Integer.valueOf(content.get(5)), 23, 59);
		dateMin = calMin.getTime();
		dateMax = calMax.getTime();
		
		priceMin = Double.valueOf(content.get(8));
		priceMax = Double.valueOf(content.get(9));
		
		updateRepertoireTable(title, genre, dateMin, dateMax, priceMin, priceMax);
		
		/////////////////////////
		
		if(currentMenu instanceof UserMenuAdmin)
		{
			UserMenuAdmin admin = (UserMenuAdmin) currentMenu;
			ArrayList<String> contentCost = admin.getAllFilterContentOfCost();
			String costType = "";
			Date costdateMin;
			Date costdateMax;
			double costpriceMin;
			double costpriceMax;
			
			if(contentCost.get(0) == "WSZYSTKIE") costType = "";
			else costType = contentCost.get(0);
			
			GregorianCalendar costcalMin = new GregorianCalendar(Integer.valueOf(contentCost.get(3)), Integer.valueOf(contentCost.get(2)) - 1, 
					Integer.valueOf(contentCost.get(1)), 0, 0);
			GregorianCalendar costcalMax = new GregorianCalendar(Integer.valueOf(contentCost.get(6)), Integer.valueOf(contentCost.get(5)) - 1, 
					Integer.valueOf(contentCost.get(4)), 23, 59);
			costdateMin = costcalMin.getTime();
			costdateMax = costcalMax.getTime();
			
			costpriceMin = Double.valueOf(contentCost.get(7));
			costpriceMax = Double.valueOf(contentCost.get(8));
			
			updateCostsTable(costType, costdateMin, costdateMax, costpriceMin, costpriceMax);
			updateFilmsTable();
		}
	}
	
	private void addSeance()
	{
		ArrayList<JComboBox> myCBs = theView.crWindowSeance.getAllComboBoxes();
		int filmIndex = myCBs.get(0).getSelectedIndex();
		String dateDay = (String)myCBs.get(1).getSelectedItem();
		String dateMonth = (String)myCBs.get(2).getSelectedItem();
		String dateYear = (String)myCBs.get(3).getSelectedItem();
		String dateHour = (String)myCBs.get(4).getSelectedItem();
		String dateMinute = (String)myCBs.get(5).getSelectedItem();
		
		GregorianCalendar myCal = new GregorianCalendar(Integer.valueOf(dateYear), Integer.valueOf(dateMonth) - 1, 
				Integer.valueOf(dateDay), Integer.valueOf(dateHour), Integer.valueOf(dateMinute));
		Date myDate = myCal.getTime();
		
		Film myFilm = theModel.repertoire.getFilm(filmIndex);
		
		Seance newSeance = new Seance(myFilm, myDate, 0);
		theModel.repertoire.add(newSeance);
		updateCosts();
		updateFiltersInView();
		
		theView.crWindowSeance.dispose();
	}
	
	private void deleteSeance()
	{
		Seance mySeance = this.sl.seance;
		if(mySeance == null) return;
		int i = 0;
		for(Seance seance : theModel.repertoire.get())
		{
			if(seance.equals(mySeance)) 
			{
				theModel.repertoire.delete(i);
				break;
			}
			i++;
		}
		updateCosts();
		updateFiltersInView();
	}
	
	private void addFilm()
	{
		ArrayList<String> content = theView.crWindowFilm.getAllContent();
		String title = content.get(0);
		String genre = content.get(1);
		double ticketPrice = Double.valueOf(content.get(2));
		double licensePrice = Double.valueOf(content.get(3));
		theModel.repertoire.addFilm(new Film(title, genre, ticketPrice, licensePrice));
		updateCosts();
		updateFiltersInView();
		theView.crWindowFilm.dispose();
	}
	
	private void deleteFilm()
	{
		Film myFilm = this.slFilms.film;
		if(myFilm == null) return;
		int i = 0;
		for(Film film : theModel.repertoire.getFilms())
		{
			if(film.equals(myFilm)) 
			{
				theModel.repertoire.deleteFilm(i);;
				break;
			}
			i++;
		}
		updateCosts();
		updateFiltersInView();
	}
	
	ActionListener userButtonListener = new ActionListener(){
		public void actionPerformed(ActionEvent e){
			theView.setVisible(false);
			theView.createUserMenu();
			currentMenu = theView.um;
			theView.um.addBuyButtonListener(buyButtonListener);
			theView.um.disableButton(theView.um.getBuyButton());
			theView.um.addBookButtonListener(bookButtonListener);
			theView.um.disableButton(theView.um.getBookButton());
			theView.um.addBackButtonListener(backButtonListener);
			theView.um.addBasketButtonListener(basketButtonListener);
			theView.um.disableButton(theView.um.getBasketButton());
			if(theModel.boughtTickets.get().isEmpty() == false)
			{
				theView.um.enableButton(theView.um.getBasketButton());
			}
			theView.um.getUserListSelection().addListSelectionListener(sl);
			updateRepertoireTable("","",null, null, 0, 0);
			
			for(JComboBox cb : theView.um.getAllComboBoxes())
			{
				cb.addActionListener(comboListener);
			}
			
			theView.um.getPriceMinTextField().addActionListener(comboListener);
			theView.um.getPriceMaxTextField().addActionListener(comboListener);
		}
	};
	
	ActionListener adminButtonListener = new ActionListener(){
		public void actionPerformed(ActionEvent e){
			theView.setVisible(false);
			theView.createAdminMenu();
			createReminder();
			//createChart(); //TEMP
			currentMenu = theView.am;
			theView.am.addBuyButtonListener(buyButtonListener);
			theView.am.addBookButtonListener(bookButtonListener);
			theView.am.addBackButtonListener(backButtonListener);
			theView.am.addBasketButtonListener(basketButtonListener);
			if(theModel.boughtTickets.get().isEmpty() == false)
			{
				theView.am.enableButton(theView.um.getBasketButton());
			}
			theView.am.getUserListSelection().addListSelectionListener(sl);
			theView.am.getCostsListSelection().addListSelectionListener(slCosts);
			theView.am.getFilmsListSelection().addListSelectionListener(slFilms);
			updateRepertoireTable("","",null, null, 0, 0);
			updateCostsTable("",null, null, 0, 0);
			updateFilmsTable();
			
			for(JComboBox cb : theView.am.getAllComboBoxes())
			{
				cb.addActionListener(comboListener);
			}
			
			for(JComboBox cb : theView.am.getAllComboBoxesOfCost())
			{
				cb.addActionListener(comboListener);
			}
			
			theView.am.getPriceMinTextField().addActionListener(comboListener);
			theView.am.getPriceMaxTextField().addActionListener(comboListener);
			theView.am.getPriceMinTextFieldOfCost().addActionListener(comboListener);
			theView.am.getPriceMaxTextFieldOfCost().addActionListener(comboListener);
			
			theView.am.getTabbedPane().addChangeListener(_currentPanelComboListener);
			
			ActionListener[] ac = { _addSeanceButtonListener, _removeSeanceButtonListener, _loadRepButtonListener, _saveRepButtonListener, 
									_chartButtonListener, _deleteButtonListener, _saveCostsButtonListener, _addFilmButtonListener, 
									_deleteFilmButtonListener};
			theView.am.addActionListenersToButtons(ac);
		}
	};
	
	ActionListener buyButtonListener = new ActionListener(){
		public void actionPerformed(ActionEvent e){
			currentMenu.createBuyTicketMenu();
			currentMenu.getUserMenu().setEnabled(false);
			currentMenu.buyTicket.getTicketCount().addChangeListener(spinnerChangeListener);
			currentMenu.buyTicket.setSeanceTitle(sl.seance.getTitle(), sl.seance.getDateAsString());
			currentMenu.buyTicket.setTicketPrice(sl.seance.getPrice());
			currentMenu.buyTicket.addBuyButtonListener(buyTicketButtonListener);
		}
	};
	
	ChangeListener spinnerChangeListener = new ChangeListener(){
		@Override
		public void stateChanged(ChangeEvent arg0) {
			double cena = sl.seance.getPrice();
			cena *= Double.valueOf(currentMenu.buyTicket.getSpinnerListModel().getValue().toString());
			currentMenu.buyTicket.setTicketPrice(cena);
		}
	};
	
	ActionListener buyTicketButtonListener = new ActionListener(){
		public void actionPerformed(ActionEvent e){
			theView.createSmallWindow("Zakup trafi³ do koszyka");
			seatPlan = sl.seance.getSeatPlan();
			for(int i = 0; i<Integer.valueOf(currentMenu.buyTicket.getSpinnerListModel().getValue().toString()); i++){
				theModel.boughtTickets.add(new Ticket(sl.seance));
//				theModel.costs.add(new Cost(new Ticket(sl.seance)));
				seatPlan++;
			}
			sl.seance.setSeatPlan(seatPlan);
			updateCosts();
			updateFiltersInView();
			currentMenu.getUserMenu().setEnabled(true);
			currentMenu.enableButton(currentMenu.getBasketButton());
		}
	};
	
	ActionListener bookButtonListener = new ActionListener(){
		public void actionPerformed(ActionEvent e){
			currentMenu.createBookTicketMenu();
			currentMenu.getUserMenu().setEnabled(false);
			currentMenu.bookTicket.getTicketCount().addChangeListener(spinnerBookChangeListener);
			currentMenu.bookTicket.setSeanceTitle(sl.seance.getTitle(), sl.seance.getDateAsString());
			currentMenu.bookTicket.setTicketPrice(sl.seance.getPrice());
			currentMenu.bookTicket.addBookButtonListener(bookTicketButtonListener);
		}
	};
	
	ChangeListener spinnerBookChangeListener = new ChangeListener(){
		@Override
		public void stateChanged(ChangeEvent arg0) {
			double cena = sl.seance.getPrice();
			cena *= Double.valueOf(currentMenu.bookTicket.getSpinnerListModel().getValue().toString());
			currentMenu.bookTicket.setTicketPrice(cena);
		}
	};
	
	ActionListener bookTicketButtonListener = new ActionListener(){
		public void actionPerformed(ActionEvent e){
			theView.createSmallWindow(String.format("<html><div style=\"width:%dpx;\">%s</div><html>",250, "Rezerwacja trafi³a do koszyka."));
			seatPlan = sl.seance.getSeatPlan();
			for(int i = 0; i<Integer.valueOf(theView.um.bookTicket.getSpinnerListModel().getValue().toString()); i++){
				theModel.reservedTickets.add(new Ticket(sl.seance));
//				theModel.costs.add(new Cost(new Ticket(sl.seance)));
				seatPlan++;
			}
			sl.seance.setSeatPlan(seatPlan);
			updateCosts();
			updateFiltersInView();
			currentMenu.getUserMenu().setEnabled(true);
			currentMenu.enableButton(currentMenu.getBasketButton());
		}
	};
	
	ActionListener backButtonListener = new ActionListener(){
		public void actionPerformed(ActionEvent e){
			theView.setVisible(true);
			currentMenu.getUserMenu().setVisible(false);
		}
	};
	
	ActionListener basketButtonListener = new ActionListener(){
		public void actionPerformed(ActionEvent e){
			currentMenu.createBasketMenu();
			currentMenu.basketMenu.getBoughtTicketsListSelection().addListSelectionListener(bought);
			currentMenu.basketMenu.getBookedTicketsListSelection().addListSelectionListener(booked);
			currentMenu.basketMenu.addDeleteTicketButtonListener(deleteTicketButtonListener);
			currentMenu.basketMenu.addDeleteReservationButtonListener(deleteReservationButtonListener);
			currentMenu.disableButton(currentMenu.basketMenu.getBoughtDeleteButton());
			currentMenu.disableButton(currentMenu.basketMenu.getBookedDeleteButton());
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
	
	ActionListener comboListener = new ActionListener(){
		public void actionPerformed(ActionEvent e){
//			JComboBox cb = (JComboBox)e.getSource();
//			String filter = (String)cb.getSelectedItem();
//			if(filter == "Gatunek"){
//				theView.um.getGenreFilterCombo().setEnabled(true);
//				
//			}
//			else theView.um.getGenreFilterCombo().setEnabled(false);
			updateFiltersInView();
		}
	};
	
	ActionListener deleteTicketButtonListener = new ActionListener(){
		public void actionPerformed(ActionEvent e){
			seatPlan = theModel.boughtTickets.get(bought.row).getSeance().getSeatPlan();
			seatPlan--;
			theModel.boughtTickets.get(bought.row).getSeance().setSeatPlan(seatPlan);
//			//
//			// USUWANIE COSTS
//			//
//			for(int i = 0; i < theModel.costs.get().size(); i++){
//				Object myObject = theModel.costs.get().get(i).getObject();
//				if(myObject instanceof Ticket){
//					System.out.println(myObject);
//					System.out.println((theModel.boughtTickets.get(bought.row)));
//					if((myObject).equals(theModel.boughtTickets.get(bought.row))){
//						theModel.costs.delete(i);
//					}
//				}
//			}
//			//
//			//
//			//
			theModel.boughtTickets.delete(bought.row);
			if(theModel.boughtTickets.get().size() == 0){
				currentMenu.basketMenu.getBasketFrame().dispose();
				theModel.boughtTickets.add(new Ticket(new Seance()));
				//theView.um.disableButton(theView.um.getBasketButton());
			}
			else{
				currentMenu.basketMenu.getBasketFrame().setVisible(false);
				updateBoughtTable();
				currentMenu.basketMenu.getBasketFrame().setVisible(true);
			}
			updateCosts();
			updateFiltersInView();
		}
	};
	
	ActionListener deleteReservationButtonListener = new ActionListener(){
		public void actionPerformed(ActionEvent e){
			seatPlan = theModel.reservedTickets.get(booked.row).getSeance().getSeatPlan();
			seatPlan--;
			theModel.reservedTickets.get(booked.row).getSeance().setSeatPlan(seatPlan);
			theModel.reservedTickets.delete(booked.row);
			if(theModel.reservedTickets.get().size() == 0){
				currentMenu.basketMenu.getBasketFrame().dispose();
				theModel.reservedTickets.add(new Ticket(new Seance()));
				//theView.um.disableButton(theView.um.getBasketButton());
			}
			else{
				currentMenu.basketMenu.getBasketFrame().setVisible(false);
				updateBookedTable();
				currentMenu.basketMenu.getBasketFrame().setVisible(true);
			}
			updateCosts();
			updateFiltersInView();
		}
	};
	
	
	///////////////////////////////////////////////////////////////////////////////////////
	
	ActionListener _addSeanceButtonListener = new ActionListener(){
		public void actionPerformed(ActionEvent e){
			theView.createCWSeance();
			theView.crWindowSeance.addActionListenersToButtons(new ActionListener[] { _OKCreationSeanceButtonListener, _cancelCreationSeanceButtonListener });
		}
	};
	
	ActionListener _removeSeanceButtonListener = new ActionListener(){
		public void actionPerformed(ActionEvent e){
			deleteSeance();
		}
	};
	
	ActionListener _loadRepButtonListener = new ActionListener(){
		public void actionPerformed(ActionEvent e){
			deserialiseRepertoire();
		}
	};
	
	ActionListener _saveRepButtonListener = new ActionListener(){
		public void actionPerformed(ActionEvent e){
			serialiseRepertoire();
		}
	};

	ActionListener _chartButtonListener = new ActionListener(){
		public void actionPerformed(ActionEvent e){
			createChart();
		}
	};
	
	ActionListener _deleteButtonListener = new ActionListener(){
		public void actionPerformed(ActionEvent e){
			clearAllByDate();
		}
	};
	
	ActionListener _saveCostsButtonListener = new ActionListener(){
		public void actionPerformed(ActionEvent e){
			saveCostsToFile();
		}
	};
	
	ActionListener _addFilmButtonListener = new ActionListener(){
		public void actionPerformed(ActionEvent e){
			theView.createCVFilm();
			theView.crWindowFilm.addActionListenersToButtons(new ActionListener[] { _OKCreationFilmButtonListener, _cancelCreationFilmButtonListener });
		}
	};
	
	ActionListener _deleteFilmButtonListener = new ActionListener(){
		public void actionPerformed(ActionEvent e){
			deleteFilm();
		}
	};
	
	ActionListener _OKCreationSeanceButtonListener = new ActionListener(){
		public void actionPerformed(ActionEvent e){
			addSeance();
		}
	};
	
	ActionListener _cancelCreationSeanceButtonListener = new ActionListener(){
		public void actionPerformed(ActionEvent e){
			theView.crWindowSeance.dispose();
		}
	};
	
	ActionListener _OKCreationFilmButtonListener = new ActionListener(){
		public void actionPerformed(ActionEvent e){
			addFilm();
		}
	};
	
	ActionListener _cancelCreationFilmButtonListener = new ActionListener(){
		public void actionPerformed(ActionEvent e){
			theView.crWindowFilm.dispose();
		}
	};
	
	ChangeListener _currentPanelComboListener = new ChangeListener(){
		@Override
		public void stateChanged(ChangeEvent arg0) {
			//Main.log("tab changed");
		}
	};
	
	//////////////////////////////////////////////////////////////////////////////////////
	
	class SelectionListener implements ListSelectionListener {
		public Seance seance;
		@Override
		public void valueChanged(ListSelectionEvent e) {
			ListSelectionModel lsm = (ListSelectionModel)e.getSource();
			// TODO Auto-generated method stub
			if(e.getValueIsAdjusting()) return;
			int row = currentMenu.getUserTable().getSelectedRow();
			if(row < 0) return;
			
			currentMenu.enableButton(currentMenu.getBuyButton());
			currentMenu.enableButton(currentMenu.getBookButton());
			seance = theModel.repertoire.get(row);
		}
	};
	
	class CostsSelectionListener implements ListSelectionListener {
		public Cost cost;
		@Override
		public void valueChanged(ListSelectionEvent e) {
			if(e.getValueIsAdjusting()) return;
			int row = ((UserMenuAdmin)currentMenu).getCostsTable().getSelectedRow();
			if(row < 0) return;
			cost = theModel.costs.get(row);
		}
	};
	
	class FilmsSelectionListener implements ListSelectionListener {
		public Film film;
		@Override
		public void valueChanged(ListSelectionEvent e) {
			if(e.getValueIsAdjusting()) return;
			int row = ((UserMenuAdmin)currentMenu).getFilmsTable().getSelectedRow();
			if(row < 0) return;
			film = theModel.repertoire.getFilms().get(row);
		}
	};
	
	class TicketsSelectionListener implements ListSelectionListener{
		public Ticket ticket;
		public int row;
		@Override
		public void valueChanged(ListSelectionEvent e) {
			// TODO Auto-generated method stub
			if(e.getValueIsAdjusting()) return;
			row = currentMenu.basketMenu.getTicketsTable().getSelectedRow();
			if(row < 0) return;
			
			if(theModel.boughtTickets.get().size() != 0)
				currentMenu.enableButton(currentMenu.basketMenu.getBoughtDeleteButton());
			System.out.println(Integer.valueOf(row));
		}
	};
	
	class BookedSelectionListener implements ListSelectionListener{
		public int row;
		@Override
		public void valueChanged(ListSelectionEvent e) {
			// TODO Auto-generated method stub
			if(e.getValueIsAdjusting()) return;
			row = currentMenu.basketMenu.getBookedTable().getSelectedRow();
			if(row < 0) return;
			
			if(theModel.reservedTickets.get().size() != 0)
				currentMenu.enableButton(currentMenu.basketMenu.getBookedDeleteButton());
			System.out.println(Integer.valueOf(row));
		}
	};
	
	public static String[] CBGetYears()
	{
		ArrayList<String> strings = new ArrayList<String>();
		GregorianCalendar cal = new GregorianCalendar();
		cal.setTime(new Date());
		int year = cal.get(GregorianCalendar.YEAR);
		for(; year >= 2000; year--)
		{
			strings.add(String.valueOf(year));
		}
		return strings.toArray(new String[] {});
	}
	
	public static String[] CBGetMonths()
	{
		ArrayList<String> strings = new ArrayList<String>();
		for(int i = 1; i <= 12; i++)
		{
			strings.add(String.valueOf(i));
		}
		return strings.toArray(new String[] {});
	}
	
	public static String[] CBGetDays()
	{
		ArrayList<String> strings = new ArrayList<String>();
		for(int i = 1; i <= 31; i++)
		{
			strings.add(String.valueOf(i));
		}
		return strings.toArray(new String[] {});
	}
	
	public static String[] CBGetHours()
	{
		ArrayList<String> strings = new ArrayList<String>();
		for(int i = 0; i <= 23; i++)
		{
			strings.add(String.valueOf(i));
		}
		return strings.toArray(new String[] {});
	}
	
	public static String[] CBGetMinutes()
	{
		ArrayList<String> strings = new ArrayList<String>();
		for(int i = 0; i <= 59; i++)
		{
			strings.add(String.valueOf(i));
		}
		return strings.toArray(new String[] {});
	}
	
	public static String[] CBGetGenres()
	{
		return new String[] {"wszystkie gatunki", "sci-fi", "krymina³", "western", "wojenny",
				"thriller", "horror", "dramat"};
	}
	
	public static String[] CBGetGenresNoAll()
	{
		return new String[] {"sci-fi", "krymina³", "western", "wojenny",
				"thriller", "horror", "dramat"};
	}
}
	
