package mainPackage.View;

import java.awt.Font;
import java.awt.event.ActionListener;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTable;
import javax.swing.ListSelectionModel;

import mainPackage.Model.TicketCollection;

public class BasketMenu extends JFrame {
	private String[] columnNames = {"Tytu³", "Data", "Cena"};
	private Object[][] content = {{"A", "B", "C"}};
	private Object[][] bookedContent = {{"A", "B", "C"}};
	private JFrame basketMenu = new JFrame();
	private JPanel basketPane = new JPanel();
	private JLabel basketLabelBought = new JLabel("Kupione bilety");
	private JLabel basketLabelBooked = new JLabel("Rezerwacje");
	private JButton boughtDeleteButton = new JButton("Usuñ");
	private JButton bookedDeleteButton = new JButton("Usuñ");
	private MyTableModel myTableModel = new MyTableModel(columnNames, content, "BoughtTicketsList");
	private MyTableModel bookedTableModel = new MyTableModel(columnNames, bookedContent, "BookedTicketsList");
	private JTable boughtTable = new JTable(myTableModel);
	private JTable bookedTable = new JTable(bookedTableModel);
	private JScrollPane scrollPane = new JScrollPane(boughtTable);
	private JScrollPane bookedScrollPane = new JScrollPane(bookedTable);
	private ListSelectionModel boughtSelectionModel = boughtTable.getSelectionModel();
	private ListSelectionModel bookedSelectionModel = bookedTable.getSelectionModel();
	
	public BasketMenu(){
		
		basketMenu.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		basketMenu.setSize(800, 600);
		
		basketPane.add(basketLabelBought);
		basketPane.add(basketLabelBooked);
		basketPane.add(scrollPane);
		basketPane.add(boughtDeleteButton);
		basketPane.add(bookedScrollPane);
		basketPane.add(bookedDeleteButton);
		basketPane.setLayout(null);
		
		basketLabelBought.setBounds(300, 10, 300, 50);
		basketLabelBought.setFont(new Font("Courier New", 2, 18));
		
		basketLabelBooked.setBounds(325, 280, 300, 50);
		basketLabelBooked.setFont(new Font("Courier New", 2, 18));
		
		scrollPane.setBounds(90, 80, 600, 125);
		bookedScrollPane.setBounds(90, 350, 600, 125);
		
		boughtDeleteButton.setBounds(330, 220, 100, 50);
		bookedDeleteButton.setBounds(330, 485, 100, 50);
		
		basketMenu.add(basketPane);
		basketMenu.setVisible(true);
	}
	
	public void addDeleteTicketButtonListener(ActionListener listenForDeleteTicketButton){
		boughtDeleteButton.addActionListener(listenForDeleteTicketButton);
	}
	
	public void addDeleteReservationButtonListener(ActionListener listenForDeleteTicketButton){
		bookedDeleteButton.addActionListener(listenForDeleteTicketButton);
	}
	
	public JFrame getBasketFrame(){
		return basketMenu;
	}
	
	public JTable getTicketsTable(){
		return boughtTable;
	}
	
	public JTable getBookedTable(){
		return bookedTable;
	}
	
	public JButton getBoughtDeleteButton(){
		return boughtDeleteButton;
	}
	
	public JButton getBookedDeleteButton(){
		return bookedDeleteButton;
	}
	
	public ListSelectionModel getBoughtTicketsListSelection(){
		return boughtSelectionModel;
	}
	
	public ListSelectionModel getBookedTicketsListSelection(){
		return bookedSelectionModel;
	}
	
	public void setBoughtTableContent(Object[][] tickets){
		myTableModel.setContent(tickets);
	}
	
	public void setBookedTableContent(Object[][] books){
		bookedTableModel.setContent(books);
	}
}
