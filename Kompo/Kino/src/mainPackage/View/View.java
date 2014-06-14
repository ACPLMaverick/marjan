package mainPackage.View;

import java.awt.BorderLayout;
import java.awt.EventQueue;
import java.awt.Font;
import java.awt.event.ActionListener;
import java.io.File;
import java.util.ArrayList;
import java.util.Date;

import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTextField;
import javax.swing.border.EmptyBorder;
import javax.swing.filechooser.FileFilter;

import mainPackage.Controller.Controller;

public class View extends JFrame {
	
	private Controller myController;
	
	private JLabel title1 = new JLabel("Witaj!");
	private JLabel title2 = new JLabel("Wybierz konto:");
	private JButton userButton = new JButton("Uzytkownik");
	private JButton adminButton = new JButton("Administrator");
	
	public UserMenuUser um;
	public UserMenuAdmin am;
	public SmallWindow window;
	
	public Chart costChart;
	
	/**
	 * Create the frame.
	 */
	public View(Controller controller) {
		this.myController = controller;
		JPanel contentPane = new JPanel();
		
		this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		this.setSize(300,280);

		contentPane.add(title1);
		contentPane.add(title2);
		contentPane.add(userButton);
		contentPane.add(adminButton);
		
		contentPane.setLayout(null);
		title1.setBounds(100, 10, 100, 50);
		title1.setFont(new Font("Courier New", 2, 20));
		title2.setBounds(60, 30, 200, 50);
		title2.setFont(new Font("Courier New", 2, 20));
		userButton.setBounds(65, 100, 150, 30);
		adminButton.setBounds(65, 150, 150, 30);
		
		this.add(contentPane);
		this.setVisible(true);
	}
	
	public void setController(Controller controller)
	{
		this.myController = controller;
	}
	
	public void addUserButtonListener(ActionListener listenForUserButton){
		userButton.addActionListener(listenForUserButton);
	}
	
	public void addAdminButtonListener(ActionListener listenForAdminButton){
		adminButton.addActionListener(listenForAdminButton);
	}
	
	public void createUserMenu(){
		if(myController == null) um = new UserMenuUser(new ArrayList<String>());
		else um = new UserMenuUser(myController.getFilmTitles());
	}
	
	public void createAdminMenu(){
		if(myController == null) am = new UserMenuAdmin(new ArrayList<String>());
		else am = new UserMenuAdmin(myController.getFilmTitles());
	}
	
	public void createSmallWindow(String txt){
		window = new SmallWindow(txt);
	}
	
	public void createSmallWindow(String txtBig, String txtSmall)
	{
		window = new SmallWindow(txtBig, txtSmall);
	}
	
	public void createCostChart(ArrayList<Number> x, ArrayList<Number> y)
	{
		try {
			costChart = new Chart("Przychody / wydatki", x, y);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	/**
	 * Metoda tworzy okno do zapisu pliku
	 * @param extension - rozszerzenie. Przyjmowane "txt" albo "xml"
	 * @return zwraca œcie¿kê do pliku
	 */
	public String createSaveMenu(String extension)
	{
		JFileChooser chooser = new JFileChooser();
		if(extension == "xml")
		{
			chooser.setFileFilter(new FileFilter()
			{

				@Override
				public boolean accept(File arg0) {
					if(arg0.isDirectory()) return true;
					return arg0.getName().endsWith(".xml");
				}

				@Override
				public String getDescription() {
					return "XML files (*.xml)";
				}
				
			});
		}
		else
		{
			chooser.setFileFilter(new FileFilter()
			{

				@Override
				public boolean accept(File arg0) {
					if(arg0.isDirectory()) return true;
					return arg0.getName().endsWith(".txt");
				}

				@Override
				public String getDescription() {
					return "Text files (*.txt)";
				}
				
			});
		}
		int control = chooser.showSaveDialog(View.this);
		if(control == JFileChooser.APPROVE_OPTION)
		{
			return chooser.getSelectedFile().getAbsolutePath() + (extension=="xml" ? ".xml" : ".txt");
		}
		else
		{
			return null;
		}
	}
}
