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

// TODO: Auto-generated Javadoc
/**
 * Klasa odpowiadaj¹ca za za³o¿enia warstwy interfejsu graficznego w architekturze MVC.
 */
public class View extends JFrame {
	
	//TODO: Okienko "o programie"
	
	private Controller myController;
	private JLabel title1 = new JLabel("Witaj!");
	private JLabel title2 = new JLabel("Wybierz konto:");
	private JButton userButton = new JButton("Uzytkownik");
	private JButton adminButton = new JButton("Administrator");
	private JButton aboutApp = new JButton("O programie");
	
	/** Referencja do menu uzytkownika. */
	public UserMenuUser um;
	
	/** Referencja do menu administratora. */
	public UserMenuAdmin am;
	
	/** Referencja do okna dialogowego. */
	public SmallWindow window;
	
	/** Referencja do okna dodawania seansow. */
	public SeanceCreationWindow crWindowSeance;
	
	/** Referencja do okna dodawania filmow. */
	public FilmCreationWindow crWindowFilm;
	
	/** Referencja do okna wykresu kosztow. */
	public Chart costChart;
	
	/**
	 * Tworzy nowy obiekt typu View spelniajacy zalozenia warstwy interfejsu graficznego architektury MVC.
	 *
	 * @param controller referencja do obiektu typu Controller, przetwarzajacej dane.
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
		contentPane.add(aboutApp);
		
		contentPane.setLayout(null);
		title1.setBounds(100, 10, 100, 50);
		title1.setFont(new Font("Courier New", 2, 20));
		title2.setBounds(60, 30, 200, 50);
		title2.setFont(new Font("Courier New", 2, 20));
		userButton.setBounds(65, 100, 150, 30);
		adminButton.setBounds(65, 150, 150, 30);
		aboutApp.setBounds(70, 190, 140, 30);
		
		this.add(contentPane);
		this.setVisible(true);
	}
	
	/**
	 * Dodaje ActionListener do guzika menu uzytkownika.
	 *
	 * @param listenForUserButton ActionListener dodawany do guzika.
	 */
	public void addUserButtonListener(ActionListener listenForUserButton){
		userButton.addActionListener(listenForUserButton);
	}
	
	/**
	 * Dodaje ActionListener do guzika menu administratora.
	 *
	 * @param listenForAdminButton ActionListener dodawany do guzika.
	 */
	public void addAdminButtonListener(ActionListener listenForAdminButton){
		adminButton.addActionListener(listenForAdminButton);
	}
	
	/**
	 * Dodaje ActionListener do guzika menu z informacjami o autorach.
	 * 
	 * @param listenForAboutAppButton ActionListener dodawany do guzika.
	 */
	public void addAboutAppButtonListener(ActionListener listenForAboutAppButton){
		aboutApp.addActionListener(listenForAboutAppButton);
	}
	
	/**
	 * Tworzy menu uzytkownika.
	 */
	public void createUserMenu(){
		if(myController == null) um = new UserMenuUser(new ArrayList<String>());
		else um = new UserMenuUser(myController.getFilmTitles());
	}
	
	/**
	 * Tworzy menu administratora.
	 */
	public void createAdminMenu(){
		if(myController == null) am = new UserMenuAdmin(new ArrayList<String>());
		else am = new UserMenuAdmin(myController.getFilmTitles());
	}
	
	/**
	 * Tworzy proste okienko z pojedynczym tekstem.
	 *
	 * @param txt tekst wyswietlany w oknie.
	 */
	public void createSmallWindow(String txt){
		window = new SmallWindow(txt);
	}
	
	/**
	 * Tworzy proste okienko z dwoma tekstami.
	 *
	 * @param txtBig tekst napisany wiekszym rozmiarem czcionki.
	 * @param txtSmall tekst napisany mniejszym rozmiarem czcionki.
	 */
	public void createSmallWindow(String txtBig, String txtSmall)
	{
		window = new SmallWindow(txtBig, txtSmall);
	}
	
	/**
	 * Tworzy proste okienko z tytulem i dwoma tekstami.
	 * 
	 * @param title tytul.
	 * @param name1 tekst napisany mniejszym rozmiarem czcionki.
	 * @param name2 tekst napisany mniejszym rozmiarem czcionki.
	 */
	public void createAboutAppWindow(String title, String name1, String name2) {
		window = new SmallWindow(title, name1, name2);
	}
	
	/**
	 * Tworzy okno dodawania seansow.
	 */
	public void createCWSeance()
	{
		this.crWindowSeance = new SeanceCreationWindow(myController.getFilmTitles(), Controller.CBGetDays(), Controller.CBGetMonths(),
												Controller.CBGetYears(), Controller.CBGetHours(), Controller.CBGetMinutes());
	}
	
	/**
	 * Tworzy okno dodawania filmow.
	 */
	public void createCVFilm()
	{
		this.crWindowFilm = new FilmCreationWindow();
	}
	
	/**
	 * Tworzy okno z wykresem kosztow.
	 *
	 * @param x wartosci na osi X
	 * @param y wartosci na osi Y
	 */
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
	 * Tworzy okno do zapisu pliku.
	 *
	 * @param extension rozszerzenie. Przyjmowane "txt" albo "xml"
	 * @return Sciezke do pliku
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
	
	/**
	 * Tworzy okno do odczytu pliku.
	 *
	 * @return Sciezke do pliku
	 */
	public String createLoadMenu()
	{
		JFileChooser chooser = new JFileChooser();
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
		int control = chooser.showOpenDialog(View.this);
		if(control == JFileChooser.APPROVE_OPTION)
		{
			return chooser.getSelectedFile().getAbsolutePath();
		}
		else
		{
			return null;
		}
	}
	
	/**
	 * Ustawia powiazanie powiazanie z klasa Controller.
	 *
	 * @param controller referencja do obiektu typu Controller, przetwarzajacej dane.
	 */
	public void setController(Controller controller)
	{
		this.myController = controller;
	}
}
