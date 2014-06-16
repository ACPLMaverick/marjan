package mainPackage.View;
import java.awt.Font;
import java.awt.event.ActionListener;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.util.ArrayList;
import java.util.Date;
import java.util.GregorianCalendar;

import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTabbedPane;
import javax.swing.JTable;
import javax.swing.JTextArea;
import javax.swing.JTextField;
import javax.swing.ListSelectionModel;
import javax.swing.table.TableColumn;

import mainPackage.Controller.Controller;


// TODO: Auto-generated Javadoc
/**
 * Klasa abstrakcyjna reprezentujaca interfejs graficzny menu uzytkownika lub administratora.
 */
public abstract class UserMenu extends JFrame {
	
	protected Date currentDate = new Date();
	protected JFrame userMenu = new JFrame();
	protected JTabbedPane tabPane = new JTabbedPane();
	protected JPanel userPane = new JPanel();
	protected JLabel userTitle = new JLabel("Witaj w koncie uzytkownika!");
	protected JLabel dateFrom = new JLabel("OD:");
	protected JLabel dateTo = new JLabel("DO:");
	protected JLabel priceFrom = new JLabel("CENA OD:");
	protected JLabel priceTo = new JLabel("CENA DO:");
	protected MyTableModel myTableModel = new MyTableModel();
	protected JTable repertoireTable = new JTable(myTableModel);
	protected ListSelectionModel selectionModel = repertoireTable.getSelectionModel();
	protected JScrollPane scrollPane = new JScrollPane(repertoireTable);
	protected JButton buyTicketButton = new JButton("Kup bilet");
	protected JButton bookTicketButton = new JButton("Rezerwuj bilet");
	protected JButton backButton = new JButton("Wstecz");
	protected JButton basketButton = new JButton("Twoje bilety");
	/*ADDED FOR FILTERING*/
	//private String[] titles = {"Gatunek", "Dzieñ", "Miesi¹c", "Rok", "Nazwa"};
	protected String[] genres = Controller.CBGetGenres();
	protected JComboBox filter;
	protected JComboBox genreFilter = new JComboBox(genres);
	protected JComboBox dayMin = new JComboBox(Controller.CBGetDays());
	protected JComboBox monthMin = new JComboBox(Controller.CBGetMonths());
	protected JComboBox yearMin = new JComboBox(Controller.CBGetYears());
	protected JComboBox dayMax = new JComboBox(Controller.CBGetDays());
	protected JComboBox monthMax = new JComboBox(Controller.CBGetMonths());
	protected JComboBox yearMax = new JComboBox(Controller.CBGetYears());
	protected JTextField priceMin = new JTextField("0.00");
	protected JTextField priceMax = new JTextField("0.00");
	
	/** Referencja do okna zakupu biletu. */
	public BuyTicketMenu buyTicket;
	
	/** Referencja do okna rezerwacji biletu. */
	public BookTicketMenu bookTicket;
	
	/** Referencja do okna menu koszyka. */
	public BasketMenu basketMenu;
	
	/**
	 * Tworzy okno menu uzytkownika lub administratora.
	 *
	 * @param filmTitles lista tytulow filmow do umieszczenia w filtrach.
	 */
	public UserMenu(ArrayList<String> filmTitles){
		filmTitles.add(0, "wszystkie filmy");
		this.filter = new JComboBox(filmTitles.toArray(new String[] {}));
		this.priceMin.addKeyListener(new MyKeyAdapter());
		this.priceMax.addKeyListener(new MyKeyAdapter());
		
		userMenu.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		userMenu.setSize(800, 600);
		
		userPane.add(scrollPane);
		userPane.add(buyTicketButton);
		userPane.add(bookTicketButton);
		userPane.add(basketButton);
		userPane.add(filter);
		userPane.add(genreFilter);
		userPane.add(dayMin);
		userPane.add(monthMin);
		userPane.add(yearMin);
		userPane.add(dayMax);
		userPane.add(monthMax);
		userPane.add(yearMax);
		userPane.add(dateFrom);
		userPane.add(dateTo);
		userPane.add(priceFrom);
		userPane.add(priceTo);
		userPane.add(priceMin);
		userPane.add(priceMax);
		userPane.setLayout(null);
		
		userMenu.add(userTitle);
		userMenu.add(backButton);
		
		userTitle.setBounds(250, 30, 300, 50);
		userTitle.setFont(new Font("Courier New", 2, 18));
		
		scrollPane.setBounds(190, 80, 580, 350);
		selectionModel.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
		TableColumn column = repertoireTable.getColumnModel().getColumn(0);
		column.setPreferredWidth(120);
		
		buyTicketButton.setBounds(150, 470, 120, 50);
		bookTicketButton.setBounds(480, 470, 120, 50);
		backButton.setBounds(30, 45, 150, 50);
		basketButton.setBounds(600, 20, 150, 50);
		
		filter.setBounds(5, 80, 178, 25);
		genreFilter.setBounds(5, 110, 178, 25);
		
		dateFrom.setFont(new Font("Courier New", 1, 13));
		dateFrom.setBounds(5, 140, 30, 25);
		
		dayMin.setBounds(35, 140, 40, 25);
		monthMin.setBounds(77, 140, 40, 25);
		yearMin.setBounds(119, 140, 65, 25);
		yearMin.setSelectedIndex(yearMin.getItemCount() - 1);
		
		dateTo.setFont(new Font("Courier New", 1, 13));
		dateTo.setBounds(5, 170, 30, 25);
		
		dayMax.setBounds(35, 170, 40, 25);
		dayMax.setSelectedIndex(dayMax.getItemCount() - 1);
		monthMax.setBounds(77, 170, 40, 25);
		monthMax.setSelectedIndex(monthMax.getItemCount() - 1);
		yearMax.setBounds(119, 170, 65, 25);
		
		priceFrom.setFont(new Font("Courier New", 1, 13));
		priceFrom.setBounds(5, 200, 80, 25);
		priceMin.setBounds(87, 200, 100, 25);
		
		priceTo.setFont(new Font("Courier New", 1, 13));
		priceTo.setBounds(5, 230, 80, 25);
		priceMax.setBounds(87, 230, 100, 25);
		
		tabPane.addTab("Repertuar", userPane);
		
		userMenu.add(tabPane);
		userMenu.setVisible(true);
	}
	
	/**
	 * Tworzy menu zakupu biletu.
	 */ 
	public void createBuyTicketMenu(){
		buyTicket = new BuyTicketMenu();
	}
	
	/**
	 * Tworzy menu rezerwacji biletu.
	 */
	public void createBookTicketMenu(){
		bookTicket = new BookTicketMenu();
	}
	
	/**
	 * Tworzy menu koszyka.
	 */
	public void createBasketMenu(){
		basketMenu = new BasketMenu();
	}
	
	/**
	 * Wylacza dzialanie guzika.
	 *
	 * @param button guzik, ktorego dzialanie ma byc wylaczone.
	 */
	public void disableButton(JButton button){
		button.setEnabled(false);
	}

	/**
	 * Wlacza dzialanie guzika.
	 *
	 * @param button guzik, ktorego dzialanie ma byc wlaczone.
	 */
	public void enableButton(JButton button){
		button.setEnabled(true);
	}
	
	/**
	 * Dodaje ActionListener do guzika zakupu biletu.
	 *
	 * @param listenForBuyButton ActionListener dodawany do guzika.
	 */
	public void addBuyButtonListener(ActionListener listenForBuyButton){
		buyTicketButton.addActionListener(listenForBuyButton);
	}
	
	/**
	 * Dodaje ActionListener do guzika rezerwacji biletu.
	 *
	 * @param listenForBookButton ActionListener dodawany do guzika.
	 */
	public void addBookButtonListener(ActionListener listenForBookButton){
		bookTicketButton.addActionListener(listenForBookButton);
	}
	
	/**
	 * Dodaje ActionListener do guzika zakupu powrotu.
	 *
	 * @param listenForBackButton ActionListener dodawany do guzika.
	 */
	public void addBackButtonListener(ActionListener listenForBackButton){
		backButton.addActionListener(listenForBackButton);
	}
	
	/**
	 * Dodaje ActionListener do guzika wyswietlenia koszyka.
	 *
	 * @param listenForBasketButton ActionListener dodawany do guzika.
	 */
	public void addBasketButtonListener(ActionListener listenForBasketButton){
		basketButton.addActionListener(listenForBasketButton);
	}
	
	/**
	 * Dodaje ActionListener do filtrowania.
	 *
	 * @param listenForComboBox ActionListener dodawany do JComboBox.
	 */
	public void addFilterComboListener(ActionListener listenForComboBox){
		filter.addActionListener(listenForComboBox);
	}
	
	/**
	 * Dodaje ActionListener do filtrowania po gatunku.
	 *
	 * @param listenForComboBox ActionListener dodawany do JComboBox.
	 */
	public void addGenreFilterComboListener(ActionListener listenForComboBox){
		genreFilter.addActionListener(listenForComboBox);
	}
	
	/**
	 * Zwraca ListSelectionModel.
	 *
	 * @return ListSelectionModel odpowiadajacy za nasluchiwanie zaznaczenia w tabeli.
	 */
	public ListSelectionModel getUserListSelection(){
		return selectionModel;
	}
	
	/**
	 * Zwraca MyTableModel.
	 *
	 * @return MyTableModel zawierajacy dane tabeli.
	 */
	public MyTableModel getMyTableModel(){
		return myTableModel;
	}
	
	/**
	 * Zwraca ramke JFrame menu uzytkownika lub administratora.
	 *
	 * @return Element interfejsu - ramke JFrame.
	 */
	public JFrame getUserMenu(){
		return userMenu;
	}
	
	/**
	 * Zwraca tabele repertuaru.
	 *
	 * @return Tabele JTable zawierajaca repertuar.
	 */
	public JTable getUserTable(){
		return repertoireTable;
	}
	
	/**
	 * Zwraca guzik do zakupu biletow.
	 *
	 * @return Guzik do zakupu biletow.
	 */
	public JButton getBuyButton(){
		return buyTicketButton;
	}
	
	/**
	 * Zwraca guzik do rezerwacji biletow.
	 *
	 * @return Guzik do rezerwacji biletow.
	 */
	public JButton getBookButton(){
		return bookTicketButton;
	}
	
	/**
	 * Zwraca guzik do menu koszyka.
	 *
	 * @return Guzik do menu koszyka.
	 */
	public JButton getBasketButton(){
		return basketButton;
	}
	
//	public JComboBox getFilterCombo(){
//		return filter;
//	}
//	
//	public JComboBox getGenreFilterCombo(){
//		return genreFilter;
//	}
	
	/**
	* Zwraca wszystkie elementy interfejsu - JComboBox.
 	*
 	* @return Wszystkie JComboBox.
 	*/
	public ArrayList<JComboBox> getAllComboBoxes()
	{
		ArrayList<JComboBox> list = new ArrayList<JComboBox>();
		list.add(filter);
		list.add(genreFilter);
		list.add(dayMin);
		list.add(monthMin);
		list.add(yearMin);
		list.add(dayMax);
		list.add(monthMax);
		list.add(yearMax);
		return list;
	}
	
	/**
	 * Zwraca cala zawartosc filtrow podana przez uzytkownka lub administratora.
	 *
	 * @return Cala zawartosc filtrow.
	 */
	public ArrayList<String> getAllFilterContent()
	{
		ArrayList<String> list = new ArrayList<String>();
		
		list.add((String)filter.getSelectedItem());
		list.add((String)genreFilter.getSelectedItem());
		list.add((String)dayMin.getSelectedItem());
		list.add((String)monthMin.getSelectedItem());
		list.add((String)yearMin.getSelectedItem());
		list.add((String)dayMax.getSelectedItem());
		list.add((String)monthMax.getSelectedItem());
		list.add((String)yearMax.getSelectedItem());
		list.add(priceMin.getText());
		list.add(priceMax.getText());
		
		return list;
	}
	
	/**
	 * Zwraca wartosc z filtru minimalnej ceny.
	 *
	 * @return Wartosc z filtru minimalnej ceny.
	 */
	public JTextField getPriceMinTextField()
	{
		return this.priceMin;
	}
	
	/**
	 * Zwraca wartosc z filtru maksymalnej ceny.
	 *
	 * @return Wartosc z filtru maksymalnej ceny.
	 */
	public JTextField getPriceMaxTextField()
	{
		return this.priceMax;
	}
	
	/**
	 * Zwraca panel zakladek.
	 *
	 * @return Panel zakladek.
	 */
	public JTabbedPane getTabbedPane() { return this.tabPane; }
	
	/**
	 * Ustawia zawartosc tabeli.
	 *
	 * @param newContent nowa zawartosc tabeli.
	 */
	public void setTableContent(Object[][] newContent) 
	{ 
		myTableModel.setContent(newContent);
		myTableModel.fireTableDataChanged();
		//System.out.println(String.valueOf(newContent.length));
	}
}
