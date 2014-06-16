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

// TODO: Auto-generated Javadoc
/**
 * Klasa reprezentujaca interfejs graficzny okna dodawania seansow.
 */
@SuppressWarnings("serial")
public class SeanceCreationWindow extends JFrame {
	
	private JPanel windowPane = new JPanel();
	private JLabel userTitle;
	private JLabel lFilm;
	private JLabel lDate;
	private JComboBox cbFilm;
	private JComboBox cbDateDay;
	private JComboBox cbDateMonth;
	private JComboBox cbDateYear;
	private JComboBox cbDateHour;
	private JComboBox cbDateMinute;
	private JButton bOK;
	private JButton bCancel;
	
	/**
	 * Tworzy nowe okno dodawania seansow z okreslonymi parametrami.
	 *
	 * @param filmTitles lista tytulow filmow.
	 * @param dateDays tablica dni.
	 * @param dateMonths tablica miesiecy.
	 * @param dateYears tablica lat.
	 * @param dateHours tablica godzin.
	 * @param dateMinutes tablica minut.
	 */
	public SeanceCreationWindow(ArrayList<String> filmTitles, String[] dateDays, String[] dateMonths, String[] dateYears, String[] dateHours, String[] dateMinutes)
	{
		userTitle = new JLabel("Nowy seans");
		bOK = new JButton("OK");
		bCancel = new JButton("Anuluj");
		lFilm = new JLabel("Film:");
		lDate = new JLabel("Data:");
		cbFilm = new JComboBox(filmTitles.toArray(new String[] {}));
		cbDateDay = new JComboBox(dateDays);
		cbDateMonth = new JComboBox(dateMonths);
		cbDateYear = new JComboBox(dateYears);
		cbDateHour = new JComboBox(dateHours);
		cbDateMinute = new JComboBox(dateMinutes);
		this.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		this.setSize(350, 200);
		
		windowPane.add(userTitle);
		windowPane.add(bOK);
		windowPane.add(bCancel);
		windowPane.add(lFilm);
		windowPane.add(lDate);
		windowPane.add(cbFilm);
		windowPane.add(cbDateDay);
		windowPane.add(cbDateMonth);
		windowPane.add(cbDateYear);
		windowPane.add(cbDateHour);
		windowPane.add(cbDateMinute);
		windowPane.setLayout(null);
		
		userTitle.setBounds(10, 0, 300, 50);
		userTitle.setFont(new Font("Courier New", 1, 16));
		bOK.setBounds(170, 130, 70, 25);
		bCancel.setBounds(245, 130, 70, 25);
		lFilm.setBounds(25, 50, 50, 25);
		cbFilm.setBounds(65, 50, 250, 25);
		lDate.setBounds(41, 85, 50, 25);
		cbDateDay.setBounds(81, 85, 40, 25);
		cbDateMonth.setBounds(123, 85, 40, 25);
		cbDateYear.setBounds(165, 85, 65, 25);
		cbDateHour.setBounds(233, 85, 40, 25);
		cbDateMinute.setBounds(274, 85, 40, 25);
		
		this.add(windowPane);
		this.setVisible(true);
	}
	
	/**
	 * Dodaje ActionListener do guzikow akceptowania i anulowania.
	 *
	 * @param col tabela ActionListener[] ktore dodawane sa do guzikow akceptowania i anulowania.
	 */
	public void addActionListenersToButtons(ActionListener[] col)
	{
		this.bOK.addActionListener(col[0]);
		this.bCancel.addActionListener(col[1]);
	}
	
	/**
	 * Zwraca cala zawartosc podana przez administratora.
	 *
	 * @return Cala zawartosc pol JComboBox.
	 */
	public ArrayList<JComboBox> getAllComboBoxes()
	{
		ArrayList<JComboBox> list = new ArrayList<JComboBox>();
		list.add(cbFilm);
		list.add(cbDateDay);
		list.add(cbDateMonth);
		list.add(cbDateYear);
		list.add(cbDateHour);
		list.add(cbDateMinute);
		return list;
	}
}
