package mainPackage.View;

import java.awt.Font;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JSpinner;
import javax.swing.JTextArea;
import javax.swing.SpinnerListModel;
import javax.swing.SwingConstants;

public class BuyTicketMenu extends TicketMenu {
	private String[] tickets = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10"}; //na sztywno
	private SpinnerListModel ticketModel = new SpinnerListModel(tickets);
	private JSpinner spinner = new JSpinner(ticketModel);
	private JLabel welcomeText = new JLabel("Kupujesz bilet na:");
	private JLabel ticketsCount = new JLabel("Iloœæ:");
	private JLabel ticketsSum = new JLabel("Suma:");
	private JButton buyButton = new JButton("Kup");
	
	public BuyTicketMenu(){
		this.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		this.setSize(300,280);
		
		this.add(welcomeText);
		this.add(ticketsCount);
		this.add(ticketsSum);
		this.add(spinner);
		this.add(buyButton);
		this.setLayout(null);
		
		welcomeText.setBounds(45, 20, 200, 50);
		welcomeText.setFont(new Font("Courier New", 2, 18));
		ticketsCount.setBounds(90, 110, 40, 20);
		ticketsSum.setBounds(85, 140, 40, 20);
		
		spinner.setBounds(150, 110, 40, 20);
		
		buyButton.setBounds(80, 180, 120, 30);
		
		this.add(getTicketMenuPane());
		this.setVisible(true);
	}
	
	public void setSeanceTitle(String text, String date){
		JLabel seanceTitle = new JLabel(text);
		JLabel seanceDate = new JLabel(date);
		this.add(seanceTitle);
		this.add(seanceDate);
		seanceTitle.setBounds(15, 40, 250, 50);
		seanceTitle.setFont(new Font("Courier New", 2, 14));
		
		seanceDate.setBounds(80, 60, 200, 50);
		seanceDate.setFont(new Font("Courier New", 2, 14));
	}
	
	public void setTicketPrice(String price){
		JTextArea sum = new JTextArea(price);
		this.add(sum);
		sum.setBounds(150,140,50,20);
		sum.setEditable(false);
	}
	
	public JSpinner getTicketCount(){
		return spinner;
	}
	
	public SpinnerListModel getSpinnerListModel(){
		return ticketModel;
	}
}
