package mainPackage.View;

import java.awt.Font;
import java.awt.event.ActionListener;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JSpinner;
import javax.swing.JTextArea;
import javax.swing.SpinnerListModel;
import javax.swing.SwingConstants;

// TODO: Auto-generated Javadoc
/**
 * Klasa reprezentujaca interfejs graficzny menu zakupu biletow dla uzytkownika lub administratora.
 */
public class BuyTicketMenu extends TicketMenu {
	
	private String[] tickets = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10"};
	private double ticketPrice;
	private SpinnerListModel ticketModel = new SpinnerListModel(tickets);
	private JSpinner spinner = new JSpinner(ticketModel);
	private JLabel welcomeText = new JLabel("Kupujesz bilet na:");
	private JLabel ticketsCount = new JLabel("Iloœæ:");
	private JLabel ticketsSum = new JLabel("Suma:");
	private JButton buyButton = new JButton("Kup");
	
	/**
	 * Tworzy okno menu zakupu biletow.
	 */
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
	
	/**
	 * Dodaje ActionListener do guzika zakupu biletu.
	 *
	 * @param buyTicketButtonListener ActionListener dodawany do guzika.
	 */
	public void addBuyButtonListener(ActionListener buyTicketButtonListener){
		buyButton.addActionListener(buyTicketButtonListener);
	}
	
	
	/**
	 * Zwraca cene biletu.
	 *
	 * @return Cene biletu.
	 */
	public double getTicketPrice(){
		return ticketPrice;
	}
	
	/**
	 * Zwraca element interfejsu JSpinner w ktorym okresla sie ilosc zakupionych biletow.
	 *
	 * @return Element interfejsu - JSpinner.
	 */
	public JSpinner getTicketCount(){
		return spinner;
	}
	
	/**
	 * Zwraca SpinnerListModel dla elementu interfejsu JSpinner.
	 *
	 * @return SpinnerListModel dla JSpinner.
	 */
	public SpinnerListModel getSpinnerListModel(){
		return ticketModel;
	}
	
	/**
	 * Ustawia tytul seansu w elemencie interfejsu JLabel.
	 *
	 * @param text tytul seansu.
	 * @param date data seansu.
	 */
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
	
	/**
	 * Ustawia cene biletu w elemencie interfejsu JTextArea.
	 *
	 * @param price cena biletu.
	 */
	public void setTicketPrice(double price){
		this.ticketPrice = price;
		JTextArea sum = new JTextArea(String.valueOf(price));
		this.add(sum);
		sum.setBounds(150,140,50,20);
		sum.setEditable(false);
	}
}
