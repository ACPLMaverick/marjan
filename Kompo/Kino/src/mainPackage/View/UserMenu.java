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
	protected String[] genres = {"wszystkie gatunki", "sci-fi", "krymina³", "western", "wojenny",
			"thriller", "horror", "dramat"};
	protected JComboBox filter;
	protected JComboBox genreFilter = new JComboBox(genres);
	protected JComboBox dayMin = new JComboBox(CBGetDays());
	protected JComboBox monthMin = new JComboBox(CBGetMonths());
	protected JComboBox yearMin = new JComboBox(CBGetYears());
	protected JComboBox dayMax = new JComboBox(CBGetDays());
	protected JComboBox monthMax = new JComboBox(CBGetMonths());
	protected JComboBox yearMax = new JComboBox(CBGetYears());
	protected JTextField priceMin = new JTextField("0.00");
	protected JTextField priceMax = new JTextField("0.00");
	
	public BuyTicketMenu buyTicket;
	public BookTicketMenu bookTicket;
	public BasketMenu basketMenu;
	
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
	
//	public JComboBox getFilterCombo(){
//		return filter;
//	}
//	
//	public JComboBox getGenreFilterCombo(){
//		return genreFilter;
//	}
	
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
	
	public JTextField getPriceMinTextField()
	{
		return this.priceMin;
	}
	
	public JTextField getPriceMaxTextField()
	{
		return this.priceMax;
	}
	
	public JTabbedPane getTabbedPane() { return this.tabPane; }
	
	public void setTableContent(Object[][] newContent) 
	{ 
		myTableModel.setContent(newContent);
		myTableModel.fireTableDataChanged();
		//System.out.println(String.valueOf(newContent.length));
	}
	
	protected String[] CBGetYears()
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
	
	protected String[] CBGetMonths()
	{
		ArrayList<String> strings = new ArrayList<String>();
		for(int i = 1; i <= 12; i++)
		{
			strings.add(String.valueOf(i));
		}
		return strings.toArray(new String[] {});
	}
	
	protected String[] CBGetDays()
	{
		ArrayList<String> strings = new ArrayList<String>();
		for(int i = 1; i <= 31; i++)
		{
			strings.add(String.valueOf(i));
		}
		return strings.toArray(new String[] {});
	}
	
	class MyKeyAdapter extends KeyAdapter {
		public void keyTyped(KeyEvent e)
		{
			char myChar = e.getKeyChar();
			if(((myChar != '0' && 
					myChar != '1' && 
					myChar != '2' &&
					myChar != '3' &&
					myChar != '4' &&
					myChar != '5' &&
					myChar != '6' &&
					myChar != '7' &&
					myChar != '8' &&
					myChar != '9' &&
					myChar != '.')) && (myChar != KeyEvent.VK_BACK_SPACE || myChar != KeyEvent.VK_ENTER))
			{
				e.consume();
			}
		}
	}
}
