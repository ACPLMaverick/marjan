package mainPackage.View;

import java.awt.Font;
import java.util.ArrayList;

import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JLabel;
import javax.swing.JPanel;

import mainPackage.Controller.Main;

public class UserMenuAdmin extends UserMenu {
	private JPanel costsPane;
	private JPanel filmsPane;
	
	private JButton addSeanceButton;
	private JButton removeSeanceButton;
	
	public UserMenuAdmin(ArrayList<String> filmTitles)
	{
		super(filmTitles);
		
		costsPane = new JPanel();
		filmsPane = new JPanel();
		addSeanceButton = new JButton("Dodaj seans");
		removeSeanceButton = new JButton("Usuñ seans");
		
		userPane.add(addSeanceButton);
		userPane.add(removeSeanceButton);
		
//		costsPane.add(secondTitle);
//		costsPane.add(switchPanesCB);
//		
//		filmsPane.add(userTitle);
//		filmsPane.add(switchPanesCB);
		
		userTitle.setText("Panel administratora");
		userTitle.setBounds(280, 30, 300, 50);
		
		addSeanceButton.setBounds(5, 260, 178, 25);
		removeSeanceButton.setBounds(5, 290, 178, 25);
		
		tabPane.add("Przychody i wydatki", costsPane);
		tabPane.add("Filmy i licencje",filmsPane);
		userPane.setVisible(true);
	}
	
	public JButton getAddSeanceButton() { return this.addSeanceButton; }
	public JButton getRemoveSeanceButton() { return this.removeSeanceButton; }
}
