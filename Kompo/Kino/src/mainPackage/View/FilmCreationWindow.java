package mainPackage.View;

import java.awt.Dimension;
import java.awt.Font;
import java.awt.event.ActionListener;
import java.util.ArrayList;

import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTextField;

import mainPackage.Controller.Controller;

@SuppressWarnings("serial")
public class FilmCreationWindow extends JFrame {
	private JPanel windowPane = new JPanel();
	private JLabel userTitle;
	private JLabel lTitle;
	private JLabel lGenre;
	private JLabel lTicketPrice;
	private JLabel lLicensePrice;
	private JTextField fTitle = new JTextField();
	private JTextField fTicketPrice = new JTextField();
	private JTextField fLicensePrice = new JTextField();
	private JComboBox cbGenre;
	private JButton bOK;
	private JButton bCancel;
	
	public FilmCreationWindow()
	{
		userTitle = new JLabel("Nowy film");
		bOK = new JButton("OK");
		bCancel = new JButton("Anuluj");
		lTitle = new JLabel("Tytu³:");
		lGenre = new JLabel("Gatunek:");
		lTicketPrice = new JLabel("Cena biletu:");
		lLicensePrice = new JLabel("Cena licencji:");
		cbGenre= new JComboBox(Controller.CBGetGenresNoAll());
		this.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		this.setSize(350, 300);
		
		windowPane.add(userTitle);
		windowPane.add(bOK);
		windowPane.add(bCancel);
		windowPane.add(lTitle);
		windowPane.add(lGenre);
		windowPane.add(lTicketPrice);
		windowPane.add(lLicensePrice);
		windowPane.add(fTitle);
		windowPane.add(fTicketPrice);
		windowPane.add(fLicensePrice);
		windowPane.add(cbGenre);
		windowPane.setLayout(null);
		
		userTitle.setBounds(10, 0, 300, 50);
		userTitle.setFont(new Font("Courier New", 1, 16));
		bOK.setBounds(170, 230, 70, 25);
		bCancel.setBounds(245, 230, 70, 25);
		
		lTitle.setBounds(10, 50, 50, 25);
		fTitle.setBounds(100, 50, 170, 25);
		
		lGenre.setBounds(10, 85, 50, 25);
		cbGenre.setBounds(100, 85, 170, 25);	
		
		lTicketPrice.setBounds(10, 120, 100, 25);
		fTicketPrice.setBounds(100, 120, 170, 25);
		fTicketPrice.addKeyListener(new MyKeyAdapter());
		
		lLicensePrice.setBounds(10, 155, 100, 25);
		fLicensePrice.setBounds(100, 155, 170, 25);
		fLicensePrice.addKeyListener(new MyKeyAdapter());
		
		this.add(windowPane);
		this.setVisible(true);
	}
	
	public ArrayList<String> getAllContent()
	{
		ArrayList<String> list = new ArrayList<String>();
		list.add(fTitle.getText());
		list.add((String)cbGenre.getSelectedItem());
		list.add(fTicketPrice.getText());
		list.add(fLicensePrice.getText());
		return list;
	}
	
	public void addActionListenersToButtons(ActionListener[] col)
	{
		this.bOK.addActionListener(col[0]);
		this.bCancel.addActionListener(col[1]);
	}
}
