package mainPackage.View;

import java.awt.Font;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTable;
import javax.swing.ListSelectionModel;

import mainPackage.Model.TicketCollection;

public class BasketMenu extends JFrame {
	private JFrame basketMenu = new JFrame();
	private JPanel basketPane = new JPanel();
	private JLabel basketLabelBought = new JLabel("Kupione bilety");
	private JLabel basketLabelBooked = new JLabel("Rezerwacje");
	private JButton basketDeleteButton = new JButton("Usuñ");
	
	public BasketMenu(Object[][] content, Object[][] bookedcontent){
		String[] columnNames = {"Tytu³", "Data", "Cena"};
		MyTableModel myTableModel = new MyTableModel(columnNames, content, "BoughtTicketsList");
		//MyTableModel bookedTableModel = new MyTableModel(columnNames, bookedcontent, "BookedTicketsList");
		JTable boughtTable = new JTable(myTableModel);
		//JTable bookedTable = new JTable(bookedTableModel);
		JScrollPane scrollPane = new JScrollPane(boughtTable);
		//JScrollPane bookedScrollPane = new JScrollPane(bookedTable);
		
		basketMenu.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		basketMenu.setSize(800, 600);
		
		basketPane.add(basketLabelBought);
		basketPane.add(basketLabelBooked);
		basketPane.add(scrollPane);
		basketPane.add(basketDeleteButton);
		//basketPane.add(bookedScrollPane);
		basketPane.setLayout(null);
		
		basketLabelBought.setBounds(300, 10, 300, 50);
		basketLabelBought.setFont(new Font("Courier New", 2, 18));
		
		basketLabelBooked.setBounds(325, 280, 300, 50);
		basketLabelBooked.setFont(new Font("Courier New", 2, 18));
		
		scrollPane.setBounds(90, 80, 600, 125);
		//bookedScrollPane.setBounds(90, 200, 600, 125);
		
		basketDeleteButton.setBounds(330, 220, 100, 50);
		
		basketMenu.add(basketPane);
		basketMenu.setVisible(true);
	}
}
