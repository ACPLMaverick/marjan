package mainPackage;
import java.awt.Font;
import java.awt.event.ActionListener;

import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTable;
import javax.swing.ListSelectionModel;
import javax.swing.table.TableColumn;


public class UserMenu extends JFrame {
	private JFrame userMenu = new JFrame();
	private JPanel userPane = new JPanel();
	private JLabel userTitle = new JLabel("Witaj w koncie uzytkownika!");
	private MyTableModel myTableModel = new MyTableModel();
	private JTable repertoireTable = new JTable(myTableModel);
	private ListSelectionModel selectionModel = repertoireTable.getSelectionModel();
	private JScrollPane scrollPane = new JScrollPane(repertoireTable);
	private JButton buyTicketButton = new JButton("Kup bilet");
	private JButton bookTicketButton = new JButton("Rezerwuj bilet");
	private JButton backButton = new JButton("Wstecz");

	
	public UserMenu(){
		userMenu.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		userMenu.setSize(800, 600);
		
		userPane.add(userTitle);
		userPane.add(scrollPane);
		userPane.add(buyTicketButton);
		userPane.add(bookTicketButton);
		userPane.add(backButton);
		userPane.setLayout(null);
		
		userTitle.setBounds(250, 10, 300, 50);
		userTitle.setFont(new Font("Courier New", 2, 18));
		
		scrollPane.setBounds(90, 80, 600, 350);
		selectionModel.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
		TableColumn column = repertoireTable.getColumnModel().getColumn(0);
		column.setPreferredWidth(120);
		
		buyTicketButton.setBounds(150, 470, 120, 50);
		bookTicketButton.setBounds(480, 470, 120, 50);
		backButton.setBounds(30, 20, 150, 50);
		
		userMenu.add(userPane);
		userMenu.setVisible(true);
	}
	
	void disableButton(JButton button){
		button.setEnabled(false);
	}

	void enableButton(JButton button){
		button.setEnabled(true);
	}
	
	void addBuyButtonListener(ActionListener listenForBuyButton){
		buyTicketButton.addActionListener(listenForBuyButton);
	}
	
	void addBookButtonListener(ActionListener listenForBookButton){
		bookTicketButton.addActionListener(listenForBookButton);
	}
	
	void addBackButtonListener(ActionListener listenForBackButton){
		backButton.addActionListener(listenForBackButton);
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
}
