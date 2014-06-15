package mainPackage.View;

import java.awt.Font;
import java.awt.event.ActionListener;
import java.util.ArrayList;

import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTabbedPane;
import javax.swing.JTable;
import javax.swing.JTextField;
import javax.swing.ListSelectionModel;
import javax.swing.table.TableColumn;

import mainPackage.Controller.Controller;
import mainPackage.Controller.Main;

public class UserMenuAdmin extends UserMenu {
	private JPanel costsPane;
	private JPanel filmsPane;
	
	private JButton addSeanceButton;
	private JButton removeSeanceButton;
	
	private JButton loadRep = new JButton("Wczytaj z XML");
	private JButton saveRep = new JButton("Zapisz do XML");
	
	////////////////////////////////////////////////
	
	protected MyTableModel costTableModel;
	protected JTable costTable;
	protected ListSelectionModel costSelectionModel;
	protected JScrollPane costScrollPane;
	
	protected JButton chartButton = new JButton("Poka¿ na wykresie");
	
	protected String[] costTypes = {"WSZYSTKIE", "LICENCJA", "SEANS", "BILET"};
	protected JComboBox costType = new JComboBox(costTypes);
	protected JComboBox costdayMin = new JComboBox(Controller.CBGetDays());
	protected JComboBox costmonthMin = new JComboBox(Controller.CBGetMonths());
	protected JComboBox costyearMin = new JComboBox(Controller.CBGetYears());
	protected JComboBox costdayMax = new JComboBox(Controller.CBGetDays());
	protected JComboBox costmonthMax = new JComboBox(Controller.CBGetMonths());
	protected JComboBox costyearMax = new JComboBox(Controller.CBGetYears());
	protected JTextField costpriceMin = new JTextField("0.00");
	protected JTextField costpriceMax = new JTextField("0.00");
	protected JLabel costdateFrom = new JLabel("OD:");
	protected JLabel costdateTo = new JLabel("DO:");
	protected JLabel costpriceFrom = new JLabel("CENA OD:");
	protected JLabel costpriceTo = new JLabel("CENA DO:");
	
	protected JButton deleteButton = new JButton("Usuñ odfiltrowane");
	
	protected JButton saveCostsButton = new JButton("Zapisz do pliku");
	
	////////////////////////////////////////////////
	
	protected MyTableModel filmsTableModel;
	protected JTable filmsTable;
	protected ListSelectionModel filmsSelectionModel;
	protected JScrollPane filmsScrollPane;
	
	protected JButton addFilmButton = new JButton("Dodaj film");
	protected JButton deleteFilmButton = new JButton("Usuñ film");
	
	public UserMenuAdmin(ArrayList<String> filmTitles)
	{
		super(filmTitles);
		
		costsPane = new JPanel();
		filmsPane = new JPanel();
		addSeanceButton = new JButton("Dodaj seans");
		removeSeanceButton = new JButton("Usuñ seans");
		costTableModel = new MyTableModel(new String[] {"Typ", "Data", "Cena"}, new Object[][] {{"","",""}}, "costs");
		costTable = new JTable(costTableModel);
		costSelectionModel = costTable.getSelectionModel();
		costScrollPane = new JScrollPane(costTable);
		filmsTableModel = new MyTableModel(new String[] {"Tytu³", "Gatunek", "Cena biletu", "Cena licencji"}, new Object[][] {{"","","", ""}}, "films");
		filmsTable = new JTable(filmsTableModel);
		filmsSelectionModel = filmsTable.getSelectionModel();
		filmsScrollPane = new JScrollPane(filmsTable);
		
		/////////////////////////////////////////////////////////////////////
		
		costScrollPane.setBounds(190, 80, 580, 350);
		costTable.setBounds(190, 80, 580, 350);
		costSelectionModel.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
		
		chartButton.setBounds(600, 20, 150, 50);
		
		costType.setBounds(5, 110, 178, 25);
		costdateFrom.setFont(new Font("Courier New", 1, 13));
		costdateFrom.setBounds(5, 140, 30, 25);
		costdayMin.setBounds(35, 140, 40, 25);
		costmonthMin.setBounds(77, 140, 40, 25);
		costyearMin.setBounds(119, 140, 65, 25);
		costyearMin.setSelectedIndex(yearMin.getItemCount() - 1);
		costdateTo.setFont(new Font("Courier New", 1, 13));
		costdateTo.setBounds(5, 170, 30, 25);
		costdayMax.setBounds(35, 170, 40, 25);
		costdayMax.setSelectedIndex(dayMax.getItemCount() - 1);
		costmonthMax.setBounds(77, 170, 40, 25);
		costmonthMax.setSelectedIndex(monthMax.getItemCount() - 1);
		costyearMax.setBounds(119, 170, 65, 25);
		costpriceFrom.setFont(new Font("Courier New", 1, 13));
		costpriceFrom.setBounds(5, 200, 80, 25);
		costpriceMin.setBounds(87, 200, 100, 25);
		costpriceTo.setFont(new Font("Courier New", 1, 13));
		costpriceTo.setBounds(5, 230, 80, 25);
		costpriceMax.setBounds(87, 230, 100, 25);
		deleteButton.setBounds(5, 260, 178, 25);
		saveCostsButton.setBounds(5, 290, 178, 25);
		
		filmsScrollPane.setBounds(190, 80, 580, 350);
		filmsTable.setBounds(190, 80, 580, 350);
		filmsSelectionModel.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
		
		addFilmButton.setBounds(5, 80, 178, 25);
		deleteFilmButton.setBounds(5, 110, 178, 25);
		
		//////////////////////////////////////////////////////////////////
		
		userPane.add(addSeanceButton);
		userPane.add(removeSeanceButton);
		userPane.add(loadRep);
		userPane.add(saveRep);
		
		costsPane.add(costScrollPane);
		costsPane.add(chartButton);
		costsPane.add(costType);
		costsPane.add(costdayMin);
		costsPane.add(costmonthMin);
		costsPane.add(costyearMin);
		costsPane.add(costdayMax);
		costsPane.add(costmonthMax);
		costsPane.add(costyearMax);
		costsPane.add(costdateFrom);
		costsPane.add(costdateTo);
		costsPane.add(costpriceFrom);
		costsPane.add(costpriceTo);
		costsPane.add(costpriceMin);
		costpriceMin.addKeyListener(new MyKeyAdapter());
		costpriceMax.addKeyListener(new MyKeyAdapter());
		costsPane.add(costpriceMax);
		costsPane.add(deleteButton);
		costsPane.add(saveCostsButton);
		costsPane.setLayout(null);
		
		filmsPane.add(filmsScrollPane);
		filmsPane.add(addFilmButton);
		filmsPane.add(deleteFilmButton);
		filmsPane.setLayout(null);
		
		//////////////////////////////////////////////////////////////////
		
		userTitle.setText("Panel administratora");
		userTitle.setBounds(280, 30, 300, 50);
		
		addSeanceButton.setBounds(5, 260, 178, 25);
		removeSeanceButton.setBounds(5, 290, 178, 25);
		loadRep.setBounds(5, 320, 178, 25);
		saveRep.setBounds(5, 350, 178, 25);
		
		tabPane.add("Przychody i wydatki", costsPane);
		tabPane.add("Filmy i licencje",filmsPane);
		userPane.setVisible(true);
	}
	
	public MyTableModel getCostsTableModel() { return this.costTableModel; }
	public MyTableModel getFilmsTableModel() { return this.filmsTableModel; }
	public ListSelectionModel getCostsListSelection() { return costSelectionModel; }
	public ListSelectionModel getFilmsListSelection() { return filmsSelectionModel; }
	public JTable getCostsTable(){ return costTable; }
	public JTable getFilmsTable(){ return filmsTable; }

	public void addActionListenersToButtons(ActionListener[] col)
	{
		this.addSeanceButton.addActionListener(col[0]);
		this.removeSeanceButton.addActionListener(col[1]);
		this.loadRep.addActionListener(col[2]);
		this.saveRep.addActionListener(col[3]);
		this.chartButton.addActionListener(col[4]);
		this.deleteButton.addActionListener(col[5]);
		this.saveCostsButton.addActionListener(col[6]);
		this.addFilmButton.addActionListener(col[7]);
		this.deleteFilmButton.addActionListener(col[8]);
	}
	
	///////////////////////////////////////////////////////////////////////////
	
	public ArrayList<JComboBox> getAllComboBoxesOfCost()
	{
		ArrayList<JComboBox> list = new ArrayList<JComboBox>();
		list.add(costType);
		list.add(costdayMin);
		list.add(costmonthMin);
		list.add(costyearMin);
		list.add(costdayMax);
		list.add(costmonthMax);
		list.add(costyearMax);
		return list;
	}
	
	public ArrayList<String> getAllFilterContentOfCost()
	{
		ArrayList<String> list = new ArrayList<String>();
		
		list.add((String)costType.getSelectedItem());
		list.add((String)costdayMin.getSelectedItem());
		list.add((String)costmonthMin.getSelectedItem());
		list.add((String)costyearMin.getSelectedItem());
		list.add((String)costdayMax.getSelectedItem());
		list.add((String)costmonthMax.getSelectedItem());
		list.add((String)costyearMax.getSelectedItem());
		list.add(costpriceMin.getText());
		list.add(costpriceMax.getText());
		
		return list;
	}
	
	public JTextField getPriceMinTextFieldOfCost()
	{
		return this.costpriceMin;
	}
	
	public JTextField getPriceMaxTextFieldOfCost()
	{
		return this.costpriceMax;
	}
	
	public void setTableContentOfCost(Object[][] newContent) 
	{ 
		costTableModel.setContent(newContent);
		costTableModel.fireTableDataChanged();
	}
	
	////////////////////////////////////////////////////////////////////////
	
	
	public void setTableContentOfFilms(Object[][] newContent) 
	{ 
		filmsTableModel.setContent(newContent);
		filmsTableModel.fireTableDataChanged();
	}
}
