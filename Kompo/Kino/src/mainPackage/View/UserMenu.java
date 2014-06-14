package mainPackage.View;
import java.awt.Font;
import java.awt.event.ActionListener;
import java.util.ArrayList;
import java.util.Date;
import java.util.GregorianCalendar;

import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTable;
import javax.swing.JTextArea;
import javax.swing.JTextField;
import javax.swing.ListSelectionModel;
import javax.swing.table.TableColumn;


public class UserMenu extends JFrame {
	private Date currentDate = new Date();
	private JFrame userMenu = new JFrame();
	private JPanel userPane = new JPanel();
	private JLabel userTitle = new JLabel("Witaj w koncie uzytkownika!");
	private JLabel dateFrom = new JLabel("OD:");
	private JLabel dateTo = new JLabel("DO:");
	private JLabel priceFrom = new JLabel("CENA OD:");
	private JLabel priceTo = new JLabel("CENA DO:");
	private MyTableModel myTableModel = new MyTableModel();
	private JTable repertoireTable = new JTable(myTableModel);
	private ListSelectionModel selectionModel = repertoireTable.getSelectionModel();
	private JScrollPane scrollPane = new JScrollPane(repertoireTable);
	private JButton buyTicketButton = new JButton("Kup bilet");
	private JButton bookTicketButton = new JButton("Rezerwuj bilet");
	private JButton backButton = new JButton("Wstecz");
	private JButton basketButton = new JButton("Twoje bilety");
	/*ADDED FOR FILTERING*/
	//private String[] titles = {"Gatunek", "Dzieñ", "Miesi¹c", "Rok", "Nazwa"};
	private String[] genres = {"wszystkie gatunki", "sci-fi", "krymina³", "western", "wojenny",
			"thriller", "horror", "dramat"};
	private JComboBox filter;
	private JComboBox genreFilter = new JComboBox(genres);
	private JComboBox dayMin = new JComboBox(CBGetDays());
	private JComboBox monthMin = new JComboBox(CBGetMonths());
	private JComboBox yearMin = new JComboBox(CBGetYears());
	private JComboBox dayMax = new JComboBox(CBGetDays());
	private JComboBox monthMax = new JComboBox(CBGetMonths());
	private JComboBox yearMax = new JComboBox(CBGetYears());
	private JTextField priceMin = new JTextField("0.00");
	private JTextField priceMax = new JTextField("0.00");
	
	public BuyTicketMenu buyTicket;
	public BookTicketMenu bookTicket;
	public BasketMenu basketMenu;
	
	public UserMenu(ArrayList<String> filmTitles){
		this.filter = new JComboBox(filmTitles.toArray(new String[] {}));
		
		userMenu.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		userMenu.setSize(800, 600);
		
		userPane.add(userTitle);
		userPane.add(scrollPane);
		userPane.add(buyTicketButton);
		userPane.add(bookTicketButton);
		userPane.add(backButton);
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
		
		userTitle.setBounds(250, 10, 300, 50);
		userTitle.setFont(new Font("Courier New", 2, 18));
		
		scrollPane.setBounds(190, 80, 580, 350);
		selectionModel.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
		TableColumn column = repertoireTable.getColumnModel().getColumn(0);
		column.setPreferredWidth(120);
		
		buyTicketButton.setBounds(150, 470, 120, 50);
		bookTicketButton.setBounds(480, 470, 120, 50);
		backButton.setBounds(30, 20, 150, 50);
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
		
		userMenu.add(userPane);
		userMenu.setVisible(true);
	}
	
	public void createBuyTicketMenu(){
		buyTicket = new BuyTicketMenu();
	}
	
	public void createBookTicketMenu(){
		bookTicket = new BookTicketMenu();
	}
	
	public void createBasketMenu(){
		basketMenu = new BasketMenu();
	}
	
	public void disableButton(JButton button){
		button.setEnabled(false);
	}

	public void enableButton(JButton button){
		button.setEnabled(true);
	}
	
	public void addBuyButtonListener(ActionListener listenForBuyButton){
		buyTicketButton.addActionListener(listenForBuyButton);
	}
	
	public void addBookButtonListener(ActionListener listenForBookButton){
		bookTicketButton.addActionListener(listenForBookButton);
	}
	
	public void addBackButtonListener(ActionListener listenForBackButton){
		backButton.addActionListener(listenForBackButton);
	}
	
	public void addBasketButtonListener(ActionListener listenForBasketButton){
		basketButton.addActionListener(listenForBasketButton);
	}
	
	public void addFilterComboListener(ActionListener listenForComboBox){
		filter.addActionListener(listenForComboBox);
	}
	
	public void addGenreFilterComboListener(ActionListener listenForComboBox){
		genreFilter.addActionListener(listenForComboBox);
	}
	
	public ListSelectionModel getUserListSelection(){
		return selectionModel;
	}
	
	public MyTableModel getMyTableModel(){
		return myTableModel;
	}
	
	public JFrame getUserMenu(){
		return userMenu;
	}
	
	public JTable getUserTable(){
		return repertoireTable;
	}
	
	public JButton getBuyButton(){
		return buyTicketButton;
	}
	
	public JButton getBookButton(){
		return bookTicketButton;
	}
	
	public JButton getBasketButton(){
		return basketButton;
	}
	
	public JComboBox getFilterCombo(){
		return filter;
	}
	
	public JComboBox getGenreFilterCombo(){
		return genreFilter;
	}
	
	public void setTableContent(Object[][] newContent) { myTableModel.setContent(newContent); }
	
	private String[] CBGetYears()
	{
		ArrayList<String> strings = new ArrayList<String>();
		GregorianCalendar cal = new GregorianCalendar();
		cal.setTime(this.currentDate);
		int year = cal.get(GregorianCalendar.YEAR);
		for(; year >= 2000; year--)
		{
			strings.add(String.valueOf(year));
		}
		return strings.toArray(new String[] {});
	}
	
	private String[] CBGetMonths()
	{
		ArrayList<String> strings = new ArrayList<String>();
		for(int i = 1; i <= 12; i++)
		{
			strings.add(String.valueOf(i));
		}
		return strings.toArray(new String[] {});
	}
	
	private String[] CBGetDays()
	{
		ArrayList<String> strings = new ArrayList<String>();
		for(int i = 1; i <= 31; i++)
		{
			strings.add(String.valueOf(i));
		}
		return strings.toArray(new String[] {});
	}
}
